import yaml
import os
from importlib import util
import importlib
import glob
import inspect
import json
import asyncio
import anyio
from mcp import ClientSession
from mcp.client.sse import sse_client
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import logging
import logging.handlers
import uuid
from datetime import datetime
import tiktoken
import pprint
import random
import string

logger = logging.getLogger(__name__)
# Add a NullHandler to the library's logger. This prevents 'No handler found'
# warnings and allows the application to configure logging.
logging.getLogger(__name__).addHandler(logging.NullHandler())

# Try to import provider libraries
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


# ------------------------------------------------- #
# 1. Data Structures & Contracts
# ------------------------------------------------- #

@dataclass
class Tool:
    name: str
    source_type: str
    source_name: str
    schema: Dict[str, Any]
    invoke: callable = None

@dataclass
class ExecutionContext:
    agent_name: str
    provider: str
    api: str  # New field: e.g., "v1_chat_completions" or "v1_responses"
    model: str
    model_settings: Dict[str, Any]
    instructions: str
    tools: List[Tool]
    policies: Dict[str, List[Dict[str, Any]]]
    history_mode: str = "full"  # Can be "full", "clean", or "none"


@dataclass
class State:
    """Represents the current state of the workflow's data."""
    # The "hot potato": the data being actively processed.
    data: Any
    original_data: Any = None # The data as it was at the start of the state.
    
    # The "context bag": holds important data from previous steps in the turn.
    context: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ActionOutput:
    next_action: Optional[str] = "continue"
    payload: Any = None
    payload_replace: bool = False
    target_agent: Optional[str] = None
    handoff_instruction: Optional[str] = None
    tool_name: Optional[str] = None
    tool_args: Optional[Dict[str, Any]] = None
    target_path: Optional[str] = None # New field for chaining custom functions
    is_guardrail: bool = False
    guardrail_result: Optional[str] = None # "passed" or "failed"

# ------------------------------------------------- #
# 2. Core Classes
# ------------------------------------------------- #

class ConfigLoader:
    @staticmethod
    def load(path: str) -> Dict[str, Any]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Configuration file not found at: {path}")
        with open(path, 'r') as f:
            content = f.read()
            for key, value in os.environ.items():
                content = content.replace(f"env:{key}", value)
            return yaml.safe_load(content)

class LLMClientManager:
    GOOGLE_GENAI_AVAILABLE = False # Class attribute

    def __init__(self, provider_configs: Dict[str, Any]):
        self.clients = {}
        self._initialize_clients(provider_configs)

    def _resolve_api_key(self, provider_name: str, provider_config: Dict[str, Any]) -> Optional[str]:
        env_var_name = f"{provider_name.upper()}_API_KEY"
        api_key = os.getenv(env_var_name)
        if api_key:
            logger.debug(f"API key for {provider_name} found in environment variable {env_var_name}.")
            return api_key

        config_api_key = provider_config.get('api_key')
        if config_api_key and isinstance(config_api_key, str):
            if config_api_key.startswith("env:"):
                env_key_from_config = config_api_key[4:]
                api_key = os.getenv(env_key_from_config)
                if api_key:
                    logger.debug(f"API key for {provider_name} resolved from config (env:{env_key_from_config}).")
                    return api_key
                else:
                    logger.warning(f"Config for {provider_name} specifies env:{env_key_from_config}, but environment variable is not set.")
            else:
                logger.debug(f"API key for {provider_name} found directly in config.")
                return config_api_key
        
        logger.warning(f"API key for {provider_name} not found in environment variables or configuration.")
        return None

    def _initialize_clients(self, provider_configs: Dict[str, Any]):
        logger.info("Initializing LLM clients...")
        if OLLAMA_AVAILABLE and 'ollama' in provider_configs:
            ollama_configs = provider_configs['ollama']
            if isinstance(ollama_configs, dict): # Handle single 'ollama' entry for backward compatibility
                ollama_configs = [ollama_configs]
            
            for config in ollama_configs:
                name = config.get('name', 'default_ollama') # Default name if not provided
                base_url = config.get('base_url')
                if not base_url:
                    logger.warning(f"Ollama configuration '{name}' is missing 'base_url'. Skipping initialization.")
                    continue
                try:
                    logger.info(f"Ollama library found. Initializing client '{name}' for {base_url}")
                    self.clients[name] = ollama.AsyncClient(host=base_url)
                except Exception as e:
                    logger.error(f"Failed to initialize Ollama client '{name}': {e}")
        else:
            logger.warning("Ollama library not found or 'ollama' not in provider configs. Using simulator for 'ollama' provider.")

        if OPENAI_AVAILABLE and 'openai' in provider_configs:
            logger.info("OpenAI library found. Initializing client.")
            openai_api_key = self._resolve_api_key('openai', provider_configs['openai'])
            if openai_api_key:
                from openai import AsyncOpenAI # Import AsyncOpenAI
                self.clients['openai'] = AsyncOpenAI(api_key=openai_api_key)
            else:
                logger.warning("OpenAI API key not found. OpenAI client not initialized.")
        else:
            logger.warning("OpenAI library not found or 'openai' not in provider configs. Using simulator for 'openai' provider.")

        # Initialize Google GenAI client
        if 'google' in provider_configs:
            try:
                from google import genai
                google_api_key = self._resolve_api_key('google', provider_configs['google'])
                if google_api_key:
                    self.clients['google'] = genai.Client(api_key=google_api_key)
                    LLMClientManager.GOOGLE_GENAI_AVAILABLE = True
                else:
                    logger.warning("Google API key not found. Google GenAI client not initialized.")
            except ImportError:
                logger.warning("Google Generative AI library not found. Using simulator for 'google' provider.")
            except Exception as e:
                logger.error(f"Failed to initialize Google GenAI client: {e}")
                

    def get_client(self, provider_name: str) -> Optional[Any]:
        return self.clients.get(provider_name)

class ToolManager:
    def __init__(self, local_tool_configs: List[Dict[str, Any]], agent_configs: List[Dict[str, Any]]):
        self.tools: Dict[str, Dict[str, Tool]] = {
            "local": {}, "mcp": {}, "handoff": {}
        }
        self._discover_local_tools(local_tool_configs)
        self.agent_configs = agent_configs

        # Generate handoff tools once for all agents at startup
        self._generate_all_handoff_tools_startup()

    def _generate_all_handoff_tools_startup(self):
        for agent_cfg in self.agent_configs:
            agent_name = agent_cfg.get("name")
            policies = agent_cfg.get("policies", {})
            # Use the new key 'handoff_as_tool_list'
            handoff_list = agent_cfg.get("handoff_as_tool_list", [])
            handoff_tools = self.generate_handoff_tools(policies, handoff_list)
            if agent_name not in self.tools["handoff"]:
                self.tools["handoff"][agent_name] = {}
            for tool in handoff_tools:
                # Store under handoff grouped by agent name and tool name
                self.tools["handoff"][agent_name][tool.name] = tool

    def get_tools_for_agent(self, agent_cfg: Dict[str,Any]) -> List[Tool]:
        tools: List[Tool] = []

        # local tools
        for req in agent_cfg.get("resources", {}).get("local_tools", []):
            grp = req["group_name"]
            tools.extend(self.tools["local"].get(grp, {}).values())

        # mcp tools
        for req in agent_cfg.get("resources", {}).get("mcp_tools", []):
            srv = req["server_name"]
            all_srv_tools = self.tools["mcp"].get(srv, {})
            flt = req.get("filter", {})
            if flt.get("permit"):
                for t in flt["permit"]:
                    if t in all_srv_tools:
                        tools.append(all_srv_tools[t])
            elif flt.get("deny"):
                for t_name, t in all_srv_tools.items():
                    if t_name not in flt["deny"]:
                        tools.append(t)
            else:
                tools.extend(all_srv_tools.values())

        # use pre-registered handoff tools for this agent from self.tools
        agent_name = agent_cfg.get("name")
        if agent_name and agent_name in self.tools["handoff"]:
            tools.extend(self.tools["handoff"][agent_name].values())
        
        return tools

    def _discover_local_tools(self, local_tool_configs: List[Dict[str, Any]]):
        logger.info("Discovering local tools...")
        for config in local_tool_configs:
            group_name = config['name']
            path_pattern = os.path.join(config['path'], '*.py')
            if group_name not in self.tools['local']:
                self.tools['local'][group_name] = {}
            for filepath in glob.glob(path_pattern):
                if '__init__' in filepath: continue
                module_name = os.path.basename(filepath)[:-3]
                module_spec = util.spec_from_file_location(module_name, filepath)
                module = util.module_from_spec(module_spec)
                try:
                    module_spec.loader.exec_module(module)
                except SyntaxError as e:
                    logger.error(f"Syntax error in tool file {filepath}: {e}")
                    continue # Skip this module and continue with others
                for name, func in inspect.getmembers(module, inspect.isfunction):
                    if hasattr(func, '_is_tool'):
                        params = inspect.signature(func).parameters
                        properties = {}
                        for p_name, p_obj in params.items():
                            p_type = 'string' # default
                            if p_obj.annotation == int:
                                p_type = 'integer'
                            elif p_obj.annotation == bool:
                                p_type = 'boolean'
                            elif p_obj.annotation == float:
                                p_type = 'number'
                            properties[p_name] = {"type": p_type}

                        schema = {
                            "type": "function",
                            "function": {
                                "name": func._tool_name,
                                "description": func._tool_description,
                                "parameters": {
                                    "type": "object",
                                    "properties": properties,
                                    "required": [p.name for p in params.values() if p.default == inspect.Parameter.empty],
                                },
                            },
                        }
                        tool = Tool(name=func._tool_name, source_type='local', source_name=group_name, schema=schema, invoke=func)
                        self.tools['local'][group_name][func._tool_name] = tool
                        logger.info(f"Discovered local tool: {group_name}.{func._tool_name}")


    def register_mcp_tools(self, server_name: str, tool_schemas: List[Any]): # tool_schemas are mcp.Tool objects
        if server_name not in self.tools['mcp']:
            self.tools['mcp'][server_name] = {}
        for mcp_tool_obj in tool_schemas: # mcp.Tool objects
            logger.debug(f"Type of MCP tool object: {type(mcp_tool_obj)}")
            logger.debug(f"Content of MCP tool object: {mcp_tool_obj}")
            
            tool_name = mcp_tool_obj.name
            tool_description = mcp_tool_obj.description
            tool_input_schema = mcp_tool_obj.inputSchema
            
            # Construct the LLM-compatible tool schema
            llm_tool_schema = {
                "type": "function",
                "function": {
                    "name": tool_name,
                    "description": tool_description,
                    "parameters": tool_input_schema if tool_input_schema else {"type": "object", "properties": {}},
                },
            }
            
            self.tools['mcp'][server_name][tool_name] = Tool(name=tool_name, source_type='mcp', source_name=server_name, schema=llm_tool_schema)
            logger.info(f"Discovered MCP tool: {server_name}.{tool_name}")
        logger.info(f"Registered {len(tool_schemas)} tools from MCP server: {server_name}")

    def generate_handoff_tools(
        self,
        policies: Dict[str, List[Dict[str, Any]]],
        handoff_defs: List[Dict[str, Any]]
    ) -> List[Tool]:
        """
        Generates handoff tools from both the agent's explicit 'handoff_as_tool_list'
        and from any handoff actions defined within its policies.
        """
        handoff_tools: List[Tool] = []
        ACTIONS = [
            "handoff", "handoff_serial", "handoff_parallel",
            "handoff_sequential", "handoff_feed_chunks"
        ]

        def make_tool_from_def(d: Dict[str, Any]) -> Optional[Tool]:
            """Factory to create a single Tool object from a handoff definition dict."""
            action = d.get("action")
            if action not in ACTIONS:
                return None

            # Use target_agent for single targets, target_agents for sequential
            tgt = None # Initialize tgt
            if action == "handoff_sequential":
                tgt_list = d.get("target_agents", [])
                if not tgt_list: return None
                tgt = "_then_".join(tgt_list) # tgt is the joined list for description
                # Now determine the name based on the new logic
                name = d.get("name") # Explicit name from config
                if not name:
                    first_agent = tgt_list[0].replace(' ', '_')
                    random_suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=6))
                    name = f"handoff_sequential_to_{first_agent}_{random_suffix}"
            else: # For non-sequential handoffs
                tgt = d.get("target_agent")
                if not tgt: return None
                name = f"{action}_to_{tgt.replace(' ', '_')}"

            # Use handoff_description for the tool's description, with a fallback
            # Now tgt is guaranteed to be set if we reach here for valid actions
            desc = d.get("handoff_description", f"Performs '{action}' handoff to '{tgt}'")
            # Use handoff_description for the tool's description, with a fallback
            desc = d.get("handoff_description", f"Performs '{action}' handoff to '{tgt}'")
            
            schema = {
              "type":"function", "function":{
                "name": name,
                "description": desc,
                "parameters": {
                  "type":"object",
                  "properties":{
                    "instruction": {
                       "type":"string",
                       "description":"What to tell the target agent"
                    },
                    "data": {
                      "type":"object",
                      "description":"Payload to hand off"
                    }
                  },
                  "required":["instruction"]
                }
              }
            }
            # store the dict `d` itself as the .invoke payload
            return Tool(
              name=name,
              source_type="handoff",
              source_name=action,
              schema=schema,
              invoke=d
            )

        # 1) Discover tools from policies
        for stage, policy_list in policies.items():
            for policy_def in policy_list:
                for action_def in policy_def.get("action_list", []):
                    tool = make_tool_from_def(action_def)
                    if tool:
                        handoff_tools.append(tool)

        # 2) Discover tools from explicit handoff_as_tool_list
        for handoff_def in handoff_defs:
            tool = make_tool_from_def(handoff_def)
            if tool:
                handoff_tools.append(tool)
        
        # De-duplicate tools by name, ensuring one definition per unique tool
        unique_tools = {t.name: t for t in handoff_tools}.values()
        return list(unique_tools)

class McpSseClient:
    """
    A lightweight SSE‐based MCP client.
    Usage:

        async with McpSseClient(url, headers) as client:
            tools = await client.list_tools()
            result = await client.call_tool('tool_id', {'foo': 'bar'})
    """
    def __init__(self,
                 url: str = "http://localhost:8505/sse",
                 headers: Optional[Dict[str, str]] = None,
                 timeout: float = 60.0): # Increased default timeout to 60 seconds
        self.url = url
        self.headers = headers
        self.timeout = timeout
        self._sse_ctx = None
        self._recv_s = None
        self._send_s = None
        self.session: ClientSession = None  # type: ignore

    async def __aenter__(self):
        # 1) open SSE connection
        self._sse_ctx = sse_client(
            url=self.url,
            headers=self.headers,
            timeout=self.timeout
        )
        self._recv_s, self._send_s = await self._sse_ctx.__aenter__()
        # 2) create MCP session and handshake
        self.session = await ClientSession(self._recv_s, self._send_s).__aenter__()
        await self.session.initialize()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        # teardown MCP session and SSE connection
        if self.session:
            await self.session.__aexit__(exc_type, exc, tb)
        if self._sse_ctx:
            await self._sse_ctx.__aexit__(exc_type, exc, tb)

    async def list_tools(self) -> Any:
        """
        Return a list of available tools.
        """
        res = await self.session.list_tools()
        return getattr(res, "tools", res)

    async def call_tool(self, tool_name: str, params: Dict[str, Any]) -> Any:
        """
        Invoke the named tool with the given params dict.
        """
        return await self.session.call_tool(tool_name, params)


class MCPManager:
    """
    Holds multiple McpSseClient instances—one per server—and
    drives their enter/exit in the same asyncio Task.
    """

    def __init__(self, servers: List[Dict[str, Any]]):
        """
        servers: list of {"name": "tools-server", "url": "http://..."} dicts
        """
        self.servers = servers
        # name -> McpSseClient
        self._clients: Dict[str, McpSseClient] = {}
        # name -> heartbeat Task
        self._heartbeat_tasks: Dict[str, asyncio.Task] = {}

    async def connect_all(self) -> None:
        """
        Open a session to every server in this same Task.
        """
        for srv in self.servers:
            name = srv["name"]
            url  = srv["url"]
            await self._connect_one(name, url)

    async def _connect_one(self, name: str, url: str) -> None:
        """
        Instantiate and __aenter__ a McpSseClient for this server.
        """
        client = McpSseClient(url=url)
        try:
            await client.__aenter__()
        except Exception as e:
            logger.error(f"[MCPManager] ERROR connecting to {name!r}: {e!r}")
            # ensure partial resources are torn down
            try:
                await client.__aexit__(None, None, None)
            except:
                pass
            return

        self._clients[name] = client
        logger.info(f"[MCPManager] Connected to {name!r}.")

        # Start heartbeat task
        self._heartbeat_tasks[name] = asyncio.create_task(self._run_heartbeat(name))

    async def _run_heartbeat(self, name: str):
        """Periodically pings a server to keep the session alive."""
        while name in self._clients:
            try:
                await asyncio.sleep(30) # 30-second interval
                client = self._clients.get(name)
                if client:
                    logger.debug(f"[MCPManager] Sending ping to {name!r}...")
                    # await client.session.ping() # Removed due to AttributeError
                    # logger.debug(f"[MCPManager] Ping to {name!r} successful.")
            except asyncio.CancelledError:
                logger.info(f"[MCPManager] Heartbeat for {name!r} cancelled.")
                break
            except Exception as e:
                logger.error(f"[MCPManager] Heartbeat for {name!r} failed: {e!r}")
                # Stop the heartbeat on persistent failure
                break

    def _get_client(self, name: str) -> McpSseClient:
        c = self._clients.get(name)
        if not c:
            raise ValueError(f"No active session for server '{name}'")
        return c

    async def list_tools(self, server_name: str) -> Any:
        """
        List tools on an already‐connected MCP server.
        """
        return await self._get_client(server_name).list_tools()

    async def invoke_tool(self, server_name: str, tool_name: str, params: Dict[str, Any]) -> Any:
        """
        Invoke a tool on an already‐connected MCP server.
        """
        return await self._get_client(server_name).call_tool(tool_name, params)

    async def disconnect(self, name: str) -> None:
        """
        Tear down the McpSseClient for `name`, in this same Task.
        """
        # Cancel heartbeat task first
        if name in self._heartbeat_tasks:
            self._heartbeat_tasks[name].cancel()
            del self._heartbeat_tasks[name]

        client = self._clients.pop(name, None)
        if not client:
            return
        try:
            await client.__aexit__(None, None, None)
        except (Exception, asyncio.CancelledError) as e:
            logger.error(f"[MCPManager] ERROR closing client for {name!r}: {e!r}")
        else:
            logger.info(f"[MCPManager] Disconnected {name!r}.")

    async def close_all(self) -> None:
        """
        Tear down *all* clients, one by one.
        """
        for name in list(self._clients):
            await self.disconnect(name)
        logger.info("[MCPManager] All sessions closed.")


class CustomActionManager:
    def invoke(self, target_function_path: str, current_data: Any) -> ActionOutput:
        logger.info(f"Invoking custom action chain starting with: {target_function_path}")
        
        next_action_str = "execute_custom_function"
        current_payload_data = current_data
        final_output = ActionOutput() # Start with a default

        while next_action_str == "execute_custom_function":
            try:
                module_path, func_name = target_function_path.rsplit('.', 1)
                module = importlib.import_module(module_path)
                func = getattr(module, func_name)
                
                action_output = func(current_payload_data)

                # --- Guardrail Logic ---
                if action_output.is_guardrail:
                    if action_output.guardrail_result == "failed":
                        logger.warning(f"Guardrail {target_function_path} failed. Halting execution and returning to user.")
                        # Override to force a return to the user with the failure message
                        return ActionOutput(
                            next_action="return_to_user",
                            payload=action_output.payload, # The failure message
                            payload_replace=True
                        )
                    else: # Passed
                        logger.info(f"Guardrail {target_function_path} passed. Continuing workflow.")
                        # Stop the custom action chain and let the main flow continue.
                        return ActionOutput(next_action="continue", payload=current_payload_data)
                
                # --- Standard Action Chaining ---
                final_output = action_output
                next_action_str = action_output.next_action or "continue"
                
                if action_output.payload_replace:
                    current_payload_data = action_output.payload
                
                target_function_path = action_output.target_path # For chaining

                if next_action_str == "execute_custom_function":
                    logger.info(f"Chaining to next custom action: {target_function_path}")

            except Exception as e:
                logger.error(f"Error invoking custom action {target_function_path}: {e}")
                # Return original data on error
                return ActionOutput(payload=current_data)

        logger.info(f"Custom action chain finished with final action: '{final_output.next_action}'")
        
        final_output.payload = current_payload_data 
        
        if final_output:
            try:
                payload_str = pprint.pformat(final_output.payload)
                logger.debug(f"Custom action final output: next_action='{final_output.next_action}', payload={payload_str}")
            except Exception as e:
                logger.debug(f"Could not format custom action payload for logging: {e}")

        return final_output


class TaruRunner:
    """
    A high-level facade for running agents.
    This class simplifies the process of initializing the framework and executing an agent.
    """
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initializes all core managers based on the provided configuration file.
        """
        self.config = ConfigLoader.load(config_path)
        self.llm_manager = LLMClientManager(self.config.get("providers", {}))
        self.tool_manager = ToolManager(self.config.get("local_tools", []), self.config.get("agents", []))
        self.mcp_manager = MCPManager(self.config.get("mcp_servers", []))
        self.custom_action_manager = CustomActionManager()
        self.agent_builder = AgentBuilder(self.config.get("agents", []), self.tool_manager)
        self.main_history: Dict[str, List[Dict[str, Any]]] = {} # Persistent history per agent
        self.response_ids: Dict[str, str] = {} # Persistent response_id per agent

        self.managers = {
            "llm": self.llm_manager,
            "tool": self.tool_manager,
            "mcp": self.mcp_manager,
            "custom_action": self.custom_action_manager,
            "builder": self.agent_builder
        }

    async def __aenter__(self):
        """Handles entering the async context, connecting MCP."""
        await self.mcp_manager.connect_all()
        for server in self.config.get("mcp_servers", []):
            name = server["name"]
            try:
                tools = await self.mcp_manager.list_tools(name)
                self.tool_manager.register_mcp_tools(name, tools)
                logging.info(f"Registered {len(tools)} tools from MCP server '{name}'")
            except Exception as e:
                log_exception(f"Failed to register tools from MCP server '{name}'", e)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Handles exiting the async context, closing MCP."""
        await self.mcp_manager.close_all()

    async def run(self, agent_name: str, user_message: List[Dict[str, Any]]) -> str:
        """
        Runs a single turn for a specified agent.

        Args:
            agent_name: The name of the agent to run (must be in config.yaml).
            user_message: The user's input message for this turn.

        Returns:
            The agent's final string response for the turn.
        """
        agent_config = next((a for a in self.config.get("agents", []) if a["name"] == agent_name), None)
        if not agent_config:
            raise ValueError(f"Agent '{agent_name}' not found in configuration.")

        execution_context = self.agent_builder.build([agent_name])
        max_turns = agent_config.get("max_turn_count", 25)
        
        # Retrieve or initialize history and response_id for this agent
        current_agent_history = self.main_history.setdefault(agent_name, [])
        last_response_id = self.response_ids.get(agent_name)

        policy_manager = PolicyManager(
            context=execution_context, 
            managers=self.managers, 
            max_turn_count=max_turns,
            initial_main_history=current_agent_history, # Pass persistent history
            initial_response_id=last_response_id # Pass the last response_id
        )
        
        final_response = await policy_manager.run(user_message)

        # Update the persistent history and response_id with the policy manager's state
        self.main_history[agent_name] = policy_manager.main_history
        if new_response_id := policy_manager.state.context.get('llm_response_id'):
            self.response_ids[agent_name] = new_response_id

        return final_response


from enum import Enum, auto

class WorkflowState(Enum):
    """Defines the explicit states of the agent's execution lifecycle."""
    AT_USER_MESSAGE = auto()
    AT_RETURN_TO_LLM = auto()
    WAITING_LLM_RESPONSE = auto() # New state
    AT_FUNCTION_CALL = auto()
    AT_FUNCTION_CALL_RETURN = auto()
    AT_LLM_FINAL_RESPONSE = auto()
    AT_USER_RETURN = auto()


class PolicyManager:
    """
    A state-driven manager for orchestrating an agent's workflow.

    This manager uses an explicit state machine to process requests,
    making the control flow transparent and easy to debug. Policies defined
    in the agent's configuration can override the default behavior at
    each state.
    """
    _PROTECTED_ACTIONS = {
        "return_to_llm", "call_llm", "process_llm_response", "call_function",
        "return_to_user", "return_user_message"
    }
    _UNCONFIGURABLE_STATES = {WorkflowState.AT_RETURN_TO_LLM, WorkflowState.WAITING_LLM_RESPONSE}
    HANDOFF_ACTIONS = {
        "handoff", "handoff_serial", "handoff_parallel",
        "handoff_sequential", "handoff_feed_chunks"
    }

    JUMP_ACTIONS = {
        "jump_to_user_message": WorkflowState.AT_USER_MESSAGE,
        "jump_to_return_to_llm": WorkflowState.AT_RETURN_TO_LLM,
        "jump_to_function_call_return": WorkflowState.AT_FUNCTION_CALL_RETURN,
        "jump_to_llm_final_response": WorkflowState.AT_LLM_FINAL_RESPONSE,
        "jump_to_user_return": WorkflowState.AT_USER_RETURN,
    }

    ALLOWED_JUMPS = {
        WorkflowState.AT_USER_MESSAGE: {WorkflowState.AT_LLM_FINAL_RESPONSE, WorkflowState.AT_USER_RETURN},
        WorkflowState.AT_RETURN_TO_LLM: {WorkflowState.AT_USER_RETURN}, # Only allowed for error handling
        WorkflowState.AT_FUNCTION_CALL: {WorkflowState.AT_FUNCTION_CALL_RETURN, WorkflowState.AT_LLM_FINAL_RESPONSE, WorkflowState.AT_USER_RETURN},
        WorkflowState.AT_FUNCTION_CALL_RETURN: {WorkflowState.AT_LLM_FINAL_RESPONSE, WorkflowState.AT_USER_RETURN},
        WorkflowState.AT_LLM_FINAL_RESPONSE: {WorkflowState.AT_RETURN_TO_LLM, WorkflowState.AT_USER_RETURN},
        WorkflowState.AT_USER_RETURN: set(),
        WorkflowState.WAITING_LLM_RESPONSE: set(), # Cannot be jumped to or from
    }

    # Combine jump and handoff actions into a single set of user-facing actions
    PUBLIC_ACTIONS = set(JUMP_ACTIONS.keys()).union(HANDOFF_ACTIONS)
    def __init__(self, context: ExecutionContext, managers: Dict[str, Any], max_turn_count: int = 25, parent_turn_info: str = "", initial_main_history: Optional[List[Dict[str, Any]]] = None, initial_response_id: Optional[str] = None):
        self.context = context
        self.managers = managers
        self.main_history: List[Dict[str, Any]] = initial_main_history if initial_main_history is not None else [] # Long-term, configurable history
        self.turn_history: List[Dict[str, Any]] = [] # Detailed log for the current turn
        self.llm_client = self.managers["llm"].get_client(self.context.provider)
        self.max_turn_count = max_turn_count
        self.current_turn_count = 0
        self.parent_turn_info = parent_turn_info
        self.visited_actions = []
        self.current_workflow_state: WorkflowState = WorkflowState.AT_USER_MESSAGE
        self.state: Optional[State] = None # Holds the data being passed between states
        self.initial_response_id = initial_response_id # Store for use in run()

    def _append_to_histories(self, message: Dict[str, Any]):
        """Appends a message to the turn history and, if applicable, the main history."""
        self.turn_history.append(message)
        
        if self.context.history_mode == "full":
            self.main_history.append(message)
        elif self.context.history_mode == "none":
            pass # Do not append to main_history
        # For "clean" mode, history is finalized at the end of the turn

    def _finalize_histories(self):
        """Finalizes the main history at the end of a turn, e.g., for 'clean' mode."""
        if self.context.history_mode == "clean" and self.turn_history:
            # Add the first user message
            if self.turn_history[0]["role"] == "user":
                self.main_history.append(self.turn_history[0])
            # Add the final assistant response
            if self.turn_history[-1]["role"] == "assistant":
                self.main_history.append(self.turn_history[-1])
        elif self.context.history_mode == "none":
            self.main_history.clear() # Ensure main_history is empty if mode is 'none'

    async def run(self, user_message: List[Dict[str, Any]]) -> str:
        """
        Executes the agent workflow as a state machine.

        The method starts at the AT_USER_MESSAGE state and loops,
        transitioning between states based on the outcome of actions,
        until it reaches the terminal AT_USER_RETURN state.
        """
        logger.info("--- Starting New State-Driven Workflow ---")
        self.turn_history = list(self.main_history) # Initialize turn_history with persistent main_history
        self.current_turn_count = 0 # Explicitly reset turn count
        self.current_workflow_state = WorkflowState.AT_USER_MESSAGE
        self.state = State(data=user_message, context={'original_user_message': user_message})

        # If a response_id was passed from a previous turn, inject it into the state.
        if self.initial_response_id:
            self.state.context['llm_response_id'] = self.initial_response_id

        while self.current_workflow_state != WorkflowState.AT_USER_RETURN:
            self.current_turn_count += 1
            if self.current_turn_count > self.max_turn_count:
                logger.error(f"Max turn count of {self.max_turn_count} exceeded.")
                error_message = (
                    f"Max actions reached ({self.max_turn_count}) before completing the request. "
                    f"Consider adjusting actions or increasing max_turn_count in the configuration. "
                    f"Visited Actions: {self.visited_actions}. "
                    f"Final Result: {self.state.data}"
                )
                return error_message

            turn_log_msg = f"Turn {self.current_turn_count}"
            if self.parent_turn_info:
                turn_log_msg = f"{self.parent_turn_info} -> Turn {self.current_turn_count}"

            logger.info(f"{turn_log_msg}: Entering state {self.current_workflow_state.name} with data: {str(self.state.data)[:100]}...")

            if self.current_workflow_state == WorkflowState.AT_USER_MESSAGE:
                await self._handle_state_at_user_message()
            elif self.current_workflow_state == WorkflowState.AT_RETURN_TO_LLM:
                await self._handle_state_at_return_to_llm()
            elif self.current_workflow_state == WorkflowState.WAITING_LLM_RESPONSE:
                await self._handle_state_waiting_llm_response()
            elif self.current_workflow_state == WorkflowState.AT_FUNCTION_CALL:
                await self._handle_state_at_function_call()
            elif self.current_workflow_state == WorkflowState.AT_FUNCTION_CALL_RETURN:
                await self._handle_state_at_function_call_return()
            elif self.current_workflow_state == WorkflowState.AT_LLM_FINAL_RESPONSE:
                await self._handle_state_at_llm_final_response()
            else:
                logger.error(f"Unknown state: {self.current_workflow_state}. Terminating.")
                self.state.data = "Error: Agent entered an unknown state."
                break

        logger.info("--- State-Driven Workflow Finished ---")
        # Write the final assistant response to history at the very end.
        if final_response := self.state.context.get('final_response'):
            self._append_to_histories({"role": "assistant", "content": [{"type": "output_text", "text": final_response}]})

        self._finalize_histories()
        await self._handle_state_at_user_return()
        return self.state.data # The final payload to the user

    # --------------------------------------------------------------------------
    # Action Resolution and Execution
    # --------------------------------------------------------------------------

    def _get_default_action_for_state(self, state: WorkflowState) -> str:
        """Returns the hardcoded default action for a given state."""
        return {
            WorkflowState.AT_USER_MESSAGE: "return_to_llm",
            WorkflowState.AT_RETURN_TO_LLM: "call_llm",
            WorkflowState.WAITING_LLM_RESPONSE: "process_llm_response",
            WorkflowState.AT_FUNCTION_CALL: "call_function",
            WorkflowState.AT_FUNCTION_CALL_RETURN: "return_to_llm",
            WorkflowState.AT_LLM_FINAL_RESPONSE: "return_to_user",
            WorkflowState.AT_USER_RETURN: "return_user_message", # Terminal action
        }.get(state, "")

    def _resolve_actions_for_state(self, state: WorkflowState, tool_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Resolves the list of actions to perform for the current state.
        It checks for a policy in the config, falling back to the default if none is found.
        Tool-specific policies are only considered for function call states.
        """
        # --- Guardrail: Prevent configuration for internal states ---
        if state in self._UNCONFIGURABLE_STATES:
            default_action = self._get_default_action_for_state(state)
            logger.debug(f"State '{state.name}' is not configurable. Using default action: '{default_action}'.")
            return [{"action": default_action}] if default_action else []

        stage_name = state.name.lower()
        policies = self.context.policies.get(stage_name, [])

        # Tool-specific policies only apply to these states
        if state in [WorkflowState.AT_FUNCTION_CALL, WorkflowState.AT_FUNCTION_CALL_RETURN]:
            for policy in policies:
                if policy.get("tool_name") == tool_name:
                    logger.debug(f"Found policy for state '{stage_name}' and tool '{tool_name}'.")
                    resolved_actions = []
                    for action_def in policy.get("action_list", []):
                        action_name = action_def.get("action")
                        if action_name in self._PROTECTED_ACTIONS:
                            logger.warning(f"User configured protected action '{action_name}' for state '{stage_name}' and tool '{tool_name}'. This is not allowed and will be ignored.")
                        else:
                            resolved_actions.append(action_def)
                    return resolved_actions
        
        # Generic policies (no tool_name specified)
        for policy in policies:
            if "tool_name" not in policy:
                logger.debug(f"Found generic policy for state '{stage_name}'.")
                resolved_actions = []
                for action_def in policy.get("action_list", []):
                    action_name = action_def.get("action")
                    if action_name in self._PROTECTED_ACTIONS:
                        logger.warning(f"User configured protected action '{action_name}' for state '{stage_name}'. This is not allowed and will be ignored.")
                    else:
                        resolved_actions.append(action_def)
                return resolved_actions

        # Fallback to default action
        default_action = self._get_default_action_for_state(state)
        logger.debug(f"No policy found for state '{stage_name}'. Using default action: '{default_action}'.")
        return [{"action": default_action}] if default_action else []

    async def action_orchestrator(self, action_list: List[Dict[str, Any]], run_default_action: bool = True, allow_state_change: bool = True) -> bool:
        """
        Orchestrates a list of actions, managing control flow between custom functions
        and standard state-machine actions. Returns True if a state change occurred.
        """
        state_changed = False
        original_state = self.current_workflow_state
        action_queue = list(action_list)

        while action_queue:
            action_config = action_queue.pop(0)
            action_name = action_config.get("action")
            if not action_name:
                continue

            logger.info(f"Orchestrating action: {action_name}")
            self.visited_actions.append(action_name)

            # --- Handle Jumps First ---
            if action_name in self.JUMP_ACTIONS:
                next_action_config = await self._handle_jump(action_config)
                if next_action_config:
                    # The jump handler returned a new action to execute (e.g., an error)
                    action_queue.insert(0, next_action_config)
                    continue
                else:
                    # The jump was executed successfully, and the state was changed.
                    # Halt the orchestrator to let the main loop continue from the new state.
                    logger.info(f"State jump to {self.current_workflow_state.name} executed. Halting orchestrator.")
                    return True

            if action_name == "execute_custom_function":
                target = action_config.get("target")
                if not target:
                    continue

                action_output = self.managers["custom_action"].invoke(target, self.state.data)

                if action_output.payload_replace:
                    self.state.data = action_output.payload

                next_action_name = action_output.next_action or "continue"
                if next_action_name != "continue":
                    action_queue.insert(0, {"action": next_action_name})
                
                continue # Continue to the next item in the queue
            
            # For standard actions, delegate to the single action executor
            state_changed = await self._execute_single_action(action_config, allow_state_change=allow_state_change)

            if state_changed:
                logger.info(f"Action '{action_name}' triggered state transition to {self.current_workflow_state.name}. Halting orchestrator.")
                return True

        # If the action list completes and the state has not changed, execute the default action if enabled.
        if run_default_action and not state_changed:
            logger.debug(f"Action queue for state {original_state.name} completed without state change. Executing default action.")
            default_action_name = self._get_default_action_for_state(original_state)
            if default_action_name:
                self.visited_actions.append(default_action_name)
                state_changed = await self._execute_single_action({"action": default_action_name}, allow_state_change=allow_state_change)
        
        return state_changed

    async def _execute_single_action(self, action_config: Dict[str, Any], allow_state_change: bool = True) -> bool:
        """
        Executes a single, non-custom action and returns True if a state change occurred.
        If allow_state_change is False, state-changing actions are logged but not executed.
        """
        action_name = action_config.get("action")
        # self.visited_actions.append(action_name) # Moved to orchestrator
        state_changed = False

        # --- State Changing Actions ---
        if action_name in ["return_to_llm", "call_llm", "call_function", "return_to_user"]:
            if not allow_state_change:
                logger.warning(f"Action '{action_name}' is a state-changing action but was called in a context that disallows it. Ignoring state change.")
                return False # Return False as no state change occurred

        if action_name == "return_to_llm":
            # This action now ONLY prepares the state for the LLM call.
            # It no longer modifies self.state.data.
            self.current_workflow_state = WorkflowState.AT_RETURN_TO_LLM
            state_changed = True
        elif action_name == "call_llm":
            # Write history at the last possible moment before the API call.
            # This logic now uses the current `state.data` as the source of truth.
            if self.state.context.get('original_user_message'):
                # It's a user turn
                # Ensure self.state.data is always a list of content blocks
                user_message_content = self.state.data
                if not isinstance(user_message_content, list):
                    user_message_content = [{"type": "input_text", "text": str(user_message_content)}]
                self._append_to_histories({"role": "user", "content": user_message_content})
                self.state.context.pop('original_user_message', None) # Clean up context
            elif self.state.context.get('tool_result'):
                # It's a function return turn
                function_call_output_message = {
                    "type": "function_call_output",
                    "call_id": self.state.context.get('tool_call_id'),
                    "name": self.state.context.get('tool_name'), # Add tool_name here
                    "output": {
                        "type": "text",
                        "text": self.state.data if isinstance(self.state.data, str) else json.dumps(self.state.data)
                    }
                }
                self._append_to_histories(function_call_output_message)
                self.state.context.pop('tool_result', None) # Clean up context

            llm_response_messages, response_id = await self._call_llm()
            self.state.context['llm_response_messages'] = llm_response_messages # Store as a list of messages
            if response_id:
                self.state.context['llm_response_id'] = response_id
            self.current_workflow_state = WorkflowState.WAITING_LLM_RESPONSE
            state_changed = True
        elif action_name == "process_llm_response":
            # History is no longer written here. It's deferred.
            llm_response_messages = self.state.context.get('llm_response_messages', [])

            # Process each message from the LLM response
            for msg in llm_response_messages:
                if msg.get("type") == "function_call":
                    self.current_workflow_state = WorkflowState.AT_FUNCTION_CALL
                    self.state.data = msg.get("arguments") # Pass arguments as data
                    self.state.context['function_call'] = msg # Store the full function_call message
                    self.state.context['tool_name'] = msg.get('name')
                    self.state.context['tool_call_id'] = msg.get('call_id') # Store call_id
                    break # Assuming only one function call per turn for now
                elif msg.get("role") == "assistant" and msg.get("content"):
                    # Concatenate content blocks for final response
                    final_content = "".join(block.get("text", "") for block in msg["content"] if block.get("type") == "output_text")
                    self.current_workflow_state = WorkflowState.AT_LLM_FINAL_RESPONSE
                    self.state.data = final_content
                    self.state.context['final_response'] = final_content
                    break # Assuming only one final text response
            state_changed = True
        elif action_name == "call_function":
            # Write the assistant's decision to call a function right before invoking it.
            if llm_fnc_call := self.state.context.get('function_call'):
                self._append_to_histories(llm_fnc_call) # This is already in the new format

            tool_name = self.state.context['tool_name']
            tool_args_str = self.state.data
            try:
                tool_args = json.loads(tool_args_str)
            except (json.JSONDecodeError, TypeError) as e:
                logger.error(f"Error decoding tool arguments JSON: {e}. Raw arguments: {tool_args_str}")
                tool_args = {} # Fallback to empty dict on error

            result = await self._invoke_tool(tool_name, tool_args)
            self.state.data = result
            self.state.context['tool_result'] = result
            
            self.current_workflow_state = WorkflowState.AT_FUNCTION_CALL_RETURN
            state_changed = True
        elif action_name == "return_to_user":
            # History is now written at the very end of the `run` method.
            self.current_workflow_state = WorkflowState.AT_USER_RETURN
            state_changed = True
        elif action_name == "return_user_message":
            # This is a terminal action, it doesn't change state, it ends the loop.
            pass

        # --- Non-State-Changing Actions ---
            # --- 1) Rewrite any handoff_* into a call_function ---
        elif action_name in self.HANDOFF_ACTIONS:
            # extract target agent (single string) or from list
            target = action_config.get("target_agent")
            if not target:
                target_list = action_config.get("target_agents", [])
                target = "".join(target_list) if target_list else ""
            
            instr       = action_config.get("handoff_instruction", "")
            context_src = action_config.get("handoff_context_source", "last_data_only")
            interval    = action_config.get("interval_seconds", None)

            # build the synthetic function name & args
            fn_name = f"{action_name}_to_{target.replace(' ', '_')}"
            fn_args = {
                "instruction":    instr,
                "context_source": context_src,
                "data": self.state.data
            }
            if interval is not None:
                fn_args["interval_seconds"] = interval

            # Here, we directly call invoke_tool, which will now handle all handoff logic.
            # The result of the handoff will be placed into self.state.data
            self.state.data = await self._invoke_tool(fn_name, fn_args)
            # Handoffs do not inherently change the FSM state, but the logic inside
            # the handoff (running a sub-agent) might. For now, we treat this action
            # as NOT changing the state of the *current* agent.
            # The history is appended within the handoff handlers.
            state_changed = False # This action itself doesn't transition the state.
            
        
        return state_changed

    async def _handle_jump(self, action_config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Handles the logic for state jumps with explicit case-by-case handling.
        """
        jump_action = action_config.get("action")
        target_state = self.JUMP_ACTIONS.get(jump_action)

        if not target_state:
            error_msg = "SW-BUG: An unknown jump action was invoked. Please contact the developer."
            logger.error(error_msg)
            return {"action": "jump_to_user_return", "data": error_msg}

        current_state = self.current_workflow_state
        logger.info(f"Attempting state jump from {current_state.name} to {target_state.name}")

        # --- AT_USER_MESSAGE Jumps ---
        if current_state == WorkflowState.AT_USER_MESSAGE:
            if target_state in self.ALLOWED_JUMPS.get(current_state, set()):
                # original_user_message should already be in the new multimodal format
                original_message_content = self.state.context.get('original_user_message', self.state.data)
                self._append_to_histories({"role": "user", "content": original_message_content})
                self.state.context['final_response'] = self.state.data
                self.current_workflow_state = target_state
                return None
            else:
                pass # Fall through to the illegal jump catch-all

        # --- AT_RETURN_TO_LLM Jumps (Error handling only) ---
        elif current_state == WorkflowState.AT_RETURN_TO_LLM:
            if target_state == WorkflowState.AT_USER_RETURN:
                self.state.context['final_response'] = self.state.data
                self.current_workflow_state = target_state
                return None
            else:
                pass # Fall through to the illegal jump catch-all

        # --- AT_FUNCTION_CALL Jumps ---
        elif current_state == WorkflowState.AT_FUNCTION_CALL:
            tool_name = self.state.context.get("tool_name", "unknown_tool")
            # Append the function_call message, which should already be in the new format
            if fnc_call_msg := self.state.context.get('function_call'):
                self._append_to_histories(fnc_call_msg)
            
            if target_state == WorkflowState.AT_FUNCTION_CALL_RETURN:
                self.state.context['tool_result'] = self.state.data
                self.current_workflow_state = target_state
                return None
            elif target_state in [WorkflowState.AT_LLM_FINAL_RESPONSE, WorkflowState.AT_USER_RETURN]:
                # A policy is skipping the function call and its return.
                # We must create a realistic history entry based on what the policy did.
                original_payload = self.state.original_data
                current_payload = self.state.data

                if original_payload != current_payload:
                    # The policy successfully provided a new result.
                    tool_output_text = json.dumps(current_payload)
                    final_response_text = str(current_payload)
                else:
                    # The policy did not provide a new result (config error).
                    # Use a safe, generic message.
                    tool_output_text = json.dumps({"status": "info", "data": "Function call did not return any data."})
                    final_response_text = "Function call did not return any data."

                # Append the synthetic tool output.
                self._append_to_histories({
                    "type": "function_call_output",
                    "call_id": self.state.context.get('tool_call_id'),
                    "name": tool_name,
                    "output": {"type": "text", "text": tool_output_text}
                })
                
                # Set the final response for the turn.
                self.state.context['final_response'] = final_response_text
                self.state.data = final_response_text # Update the hot potato
                self.current_workflow_state = target_state
                return None
            else:
                pass # Fall through to the illegal jump catch-all

        # --- AT_FUNCTION_CALL_RETURN Jumps ---
        elif current_state == WorkflowState.AT_FUNCTION_CALL_RETURN:
            tool_name = self.state.context.get("tool_name", "unknown_tool")
            # If skipping function call return, append a placeholder output
            if target_state in self.ALLOWED_JUMPS.get(current_state, set()):
                self._append_to_histories({
                    "type": "function_call_output",
                    "call_id": self.state.context.get('tool_call_id'),
                    "name": tool_name,
                    "output": {"type": "text", "text": "Function call result skipped by policy."}
                })
                # To mimic the natural flow, we must set final_response before entering
                # either AT_LLM_FINAL_RESPONSE or the terminal AT_USER_RETURN state.
                self.state.context['final_response'] = self.state.data
                self.current_workflow_state = target_state
                return None
            else:
                pass # Fall through to the illegal jump catch-all

        # --- AT_LLM_FINAL_RESPONSE Jumps ---
        elif current_state == WorkflowState.AT_LLM_FINAL_RESPONSE:
            # If a jump occurs from this state, it means the agent's final response
            # (stored in context['final_response']) is being intercepted by a policy.
            # We must log this final response to history before proceeding with the jump.
            if final_response := self.state.context.get('final_response'):
                self._append_to_histories({"role": "assistant", "content": [{"type": "output_text", "text": final_response}]})

            if target_state == WorkflowState.AT_RETURN_TO_LLM:
                self.state.context.pop('final_response', None)
                self.state.context['original_user_message'] = self.state.data
                self.current_workflow_state = target_state
                return None
            elif target_state == WorkflowState.AT_USER_RETURN:
                # The final_response is already set and has been logged.
                self.current_workflow_state = target_state
                return None
            else:
                pass # Fall through to the illegal jump catch-all
        else: # This else now catches any unhandled current_state
            pass # Fall through to the illegal jump catch-all

        # --- Illegal Jump Catch-All ---
        error_msg = f"Illegal state jump requested. From: {current_state.name}, To: {target_state.name}."
        logger.error(error_msg)
        return {"action": "jump_to_user_return", "data": error_msg}
    

    async def _handle_state_at_user_message(self):
        logger.debug("Handling state: AT_USER_MESSAGE")
        self.state.original_data = self.state.data
        # self._append_to_histories({"role": "user", "content": self.state.data}) # Deferred
        action_list = self._resolve_actions_for_state(self.current_workflow_state)
        await self.action_orchestrator(action_list)

    async def _handle_state_at_return_to_llm(self):
        logger.debug("Handling state: AT_RETURN_TO_LLM")
        self.state.original_data = self.state.data
        action_list = self._resolve_actions_for_state(self.current_workflow_state)
        await self.action_orchestrator(action_list)

    async def _handle_state_waiting_llm_response(self):
        logger.debug("Handling state: WAITING_LLM_RESPONSE")
        self.state.original_data = self.state.data
        action_list = self._resolve_actions_for_state(self.current_workflow_state)
        await self.action_orchestrator(action_list)

    async def _handle_state_at_function_call(self):
        logger.debug("Handling state: AT_FUNCTION_CALL")
        self.state.original_data = self.state.data
        tool_name = self.state.context.get("tool_name")
        action_list = self._resolve_actions_for_state(self.current_workflow_state, tool_name=tool_name)
        await self.action_orchestrator(action_list)

    async def _handle_state_at_function_call_return(self):
        logger.debug("Handling state: AT_FUNCTION_CALL_RETURN")
        self.state.original_data = self.state.data
        tool_name = self.state.context.get("tool_name")

        # Resolve and execute the action list from the policy.
        action_list = self._resolve_actions_for_state(self.current_workflow_state, tool_name=tool_name)
        
        # The orchestrator will handle custom functions and the default action.
        # History is now written by the `return_to_llm` action.
        await self.action_orchestrator(action_list)

    async def _handle_state_at_llm_final_response(self):
        logger.debug("Handling state: AT_LLM_FINAL_RESPONSE")
        self.state.original_data = self.state.data
        action_list = self._resolve_actions_for_state(self.current_workflow_state)
        await self.action_orchestrator(action_list)

    async def _handle_state_at_user_return(self):
        logger.debug("Handling state: AT_USER_RETURN")
        self.state.original_data = self.state.data
        action_list = self._resolve_actions_for_state(self.current_workflow_state)
        # In this final state, we execute actions but ignore any that would change the state.
        await self.action_orchestrator(action_list, run_default_action=False, allow_state_change=False)

    # --------------------------------------------------------------------------
    # Ported Logic (from original PolicyManager)
    # --------------------------------------------------------------------------

    async def _call_llm(self) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        """Calls the configured LLM and returns messages and the response ID."""
        all_tools = self.context.tools
        previous_response_id = self.state.context.get('llm_response_id')
        history_mode = self.context.history_mode

        try:
            if self.context.api == 'v1_responses':
                from .adapters.openai_responses_adapter import call_responses_api
                messages, response_id = await call_responses_api(
                    client=self.llm_client,
                    model=self.context.model,
                    messages=self.turn_history,
                    tools=all_tools,
                    instructions=self.context.instructions,
                    model_settings=self.context.model_settings,
                    previous_response_id=previous_response_id,
                    history_mode=history_mode
                )
                return messages, response_id

            elif self.context.api == 'v1_chat_completions':
                from .adapters.openai_chat_completions_adapter import call_chat_completions_api
                messages, response_id = await call_chat_completions_api(
                    client=self.llm_client,
                    model=self.context.model,
                    messages=self.turn_history,
                    tools=all_tools,
                    instructions=self.context.instructions,
                    model_settings=self.context.model_settings,
                    history_mode=history_mode
                )
                return messages, response_id

            # Handling for other APIs that might not return a response_id
            elif self.context.api == 'ollama_openai_compatible':
                from .adapters.ollama_adapter import call_ollama_api
                messages = await call_ollama_api(
                    client=self.llm_client,
                    model=self.context.model,
                    messages=self.turn_history,
                    tools=all_tools,
                    instructions=self.context.instructions,
                    model_settings=self.context.model_settings,
                    history_mode=history_mode
                )
                return messages, None # No response_id for ollama yet

            elif self.context.api == 'gemini_v1beta':
                from .adapters.gemini_adapter import call_gemini_api
                messages = await call_gemini_api(
                    client=self.llm_client,
                    model=self.context.model,
                    messages=self.turn_history,
                    tools=all_tools,
                    instructions=self.context.instructions,
                    model_settings=self.context.model_settings,
                    history_mode=history_mode
                )
                return messages, None # No response_id for gemini yet

            else:
                logger.error(f"Unknown API type specified: {self.context.api}")
                error_message = [{"role": "assistant", "content": [{"type": "output_text", "text": f"Error: Unknown API type '{self.context.api}'"}]}]
                return error_message, None

        except Exception as e:
            log_exception("Calling LLM", e)
            error_message = [{"role": "assistant", "content": [{"type": "output_text", "text": f"Error: The language model call failed with error: {e}"}]}]
            return error_message, None

    async def _invoke_tool(self, tool_name: str, tool_args: dict) -> Optional[Any]:
        """
        1) Invoke the local or MCP tool
        2) Normalize its output
        3) On error/empty, append a failure message and return None
        4) On success, return the normalized result.
        (Appending to history is now handled by the main run loop)
        """
        try:
            # locate the tool
            tool = next((t for t in self.context.tools if t.name == tool_name), None)
            if not tool:
                raise ValueError(f"Tool '{tool_name}' not in execution context.")

            # invoke
            if tool.source_type == "local":
                result = tool.invoke(**tool_args)
            elif tool.source_type == "handoff":
                # 1. Parse "handoff[_flavor]_to_<agent>"
                parts = tool.name.split("_to_", 1)
                if len(parts) != 2:
                    raise ValueError(f"Invalid handoff tool name format: {tool.name}")
                
                handoff_flavor, target_agent = parts
                
                # 2. Prepare arguments for the handler
                # The policy dict from the tool definition
                policy = tool.invoke 
                # The arguments passed in the tool call (e.g., from LLM or _execute_single_action)
                state_args = tool_args 

                # 3. Dispatch to the correct handler based on flavor
                if handoff_flavor == "handoff":
                    return await self._handle_handoff(policy, state_args, tool.name)
                elif handoff_flavor == "handoff_serial":
                    return await self._handle_serial_handoff(policy, state_args, tool.name)
                elif handoff_flavor == "handoff_parallel":
                    return await self._handle_parallel_handoff(policy, state_args, tool.name)
                elif handoff_flavor == "handoff_sequential":
                    return await self._handle_sequential_handoff(policy, state_args, tool.name)
                elif handoff_flavor == "handoff_feed_chunks":
                    return await self._handle_handoff_feed_chunks(policy, state_args, tool.name)
                else:
                    raise NotImplementedError(f"Handoff flavor '{handoff_flavor}' is not implemented.")
            
            else:
                # 'mcp' source_type or any other types (except 'custom' which is removed)
                result = await self.managers["mcp"].invoke_tool(
                    tool.source_name, tool_name, tool_args
                )
        except Exception as e:
            log_exception(f"Invoking tool '{tool_name}'", e)
            # Return an error dictionary instead of writing to history
            return {"status": "error", "data": f"Failed to invoke tool '{tool_name}': {e}"}

        # raw debug
        raw_debug = pprint.pformat(result, indent=2)
        logging.debug(f"Raw tool_result for '{tool_name}':\n{raw_debug}")

        # normalize call tool wrapper & text chunks
        try:
            # The result from MCP is often a CallToolResult object.
            # We need to extract the actual data from its content.
            if hasattr(result, "content") and result.content:
                content = result.content
                # The content is usually a list of TextContent objects.
                if isinstance(content, list) and content and hasattr(content[0], "text"):
                    # We join the text parts, which often form a single JSON string.
                    json_string = "".join(item.text for item in content)
                    try:
                        # We parse the JSON string to get the actual data.
                        result = json.loads(json_string)
                    except (json.JSONDecodeError, TypeError):
                        # If it's not valid JSON, we just use the text content.
                        result = json_string
        except Exception as e:
            # If any part of the normalization fails, log it and proceed with the raw result.
            logging.debug(f"Could not normalize MCP result, proceeding with raw data. Error: {e}")

        logging.debug(f"Normalized tool_result for '{tool_name}': {pprint.pformat(result)}")

        # empty / falsy check
        if result is None or (isinstance(result, (str, list, dict)) and not result):
            logging.error(f"No data returned from tool '{tool_name}': {result!r}")
            # Return an error dictionary
            return {"status": "error", "data": f"No data returned from tool '{tool_name}'."}

        # MCP style error dict? (This check remains valid)
        if isinstance(result, dict) and result.get("status") == "error":
            # The tool itself reported an error, so we just pass it through.
            return result

        # success → return result for run loop to handle
        return result


    def _get_last_user_message(self) -> Optional[str]:
        """Retrieves the content of the most recent user message from the turn history."""
        for message in reversed(self.turn_history):
            if message.get("role") == "user" and message.get("content"):
                # Assuming content is a list of content blocks, extract text from the first one
                if isinstance(message["content"], list) and message["content"] and message["content"][0].get("type") == "input_text":
                    return message["content"][0].get("text")
        return None
    
    

    def _format_data_to_string(self, d: Any) -> str:
        """Intelligently formats data into a natural language string."""
        if isinstance(d, str):
            return d
        if isinstance(d, dict):
            # Format dict as "key1: value1, key2: value2"
            return ", ".join([f"{k}: {v}" for k, v in d.items()])
        try:
            # Fallback for other types like lists or complex objects
            return json.dumps(d)
        except (TypeError, OverflowError):
            return str(d)

    def _build_handoff_message(self, context_source: str, data: Any, chunk_num: Optional[int] = None) -> Dict[str, Any]:
        """
        Builds the initial user message for a handoff based on the context source.
        The handoff instruction is handled separately as the system prompt and is not part of this message.
        Returns the message in the new internal format.
        """
        final_prompt_text = ""
        
        # This case remains untouched for now, as requested.
        if context_source == "turn_history":
            data_label = "Data Chunk" if chunk_num is not None else "Data"
            formatted_data = self._format_data_to_string(data)
            final_prompt_text = f"The following data was produced by the previous agent:\n{formatted_data}"

        else:
            # --- New Combined Logic ---
            original_user_message_content = self.state.context.get('original_user_message', [])
            
            # Extract text from the original user message content blocks
            user_message_text = ""
            if isinstance(original_user_message_content, list):
                text_parts = [block.get("text", "") for block in original_user_message_content if block.get("type") == "input_text"]
                user_message_text = "".join(text_parts)
            elif isinstance(original_user_message_content, str): # Fallback for old format
                user_message_text = original_user_message_content

            # Format the current data payload into a string
            data_text = self._format_data_to_string(data)

            # Construct the final prompt based on the context source
            if context_source == "user_message_and_data":
                # Combine original message and current data into one string
                final_prompt_text = f"{user_message_text}, {data_text}"
            elif context_source == "user_message_only":
                final_prompt_text = user_message_text
            else: # Default to "last_data_only"
                # Use only the formatted data as the prompt, no wrapper
                final_prompt_text = data_text

        return {"role": "user", "content": [{"type": "input_text", "text": final_prompt_text}]}

    async def _handle_handoff(self, policy: Dict[str, Any], state_args: Dict[str, Any], original_tool_name: str):
        target_agent_name = policy.get("target_agent")
        # The handoff's system-level instruction is sourced from the policy configuration.
        # If not found, the target agent's own default instruction will be used.
        handoff_instruction = policy.get("handoff_instruction")
        handoff_context_source = policy.get("handoff_context_source", None) # Default to None if not present
        handoff_data = state_args.get("data", self.state.data)

        logger.info(f"Handoff to agent: {target_agent_name} with context source: {handoff_context_source or 'LLM-driven'}")
        
        handoff_context = self.managers["builder"].build([target_agent_name])
        
        # Override the target agent's instructions if a handoff_instruction is provided
        if handoff_instruction:
            logger.info(f"Overriding target agent instruction with: {handoff_instruction[:100]}...")
            handoff_context.instructions = handoff_instruction

        remaining_turns = self.max_turn_count - self.current_turn_count
        if remaining_turns < 1:
            logger.warning(f"Not enough turns left for handoff to agent {target_agent_name}.")
            # This part remains the same
            return

        parent_turn_info = f"Handoff from {self.context.agent_name} (Turn {self.current_turn_count}/{self.max_turn_count})"
        
        # --- Message Construction Logic ---
        handoff_message = None
        initial_history = None
        if handoff_context_source:
            # Case 1: Programmatic handoff (from action)
            if handoff_context_source == "turn_history":
                initial_history = list(self.turn_history)
            handoff_message = self._build_handoff_message(handoff_context_source, handoff_data)
        else:
            # Case 2: LLM-driven handoff (from tool)
            instruction = state_args.get("instruction", "")
            data_str = self._format_data_to_string(handoff_data)
            # Combine instruction and data from the tool call into a single prompt
            combined_prompt = f"{instruction}, {data_str}" if instruction and data_str else instruction or data_str
            handoff_message = {"role": "user", "content": [{"type": "input_text", "text": combined_prompt}]}

        handoff_manager = PolicyManager(
            context=handoff_context, 
            managers=self.managers, 
            max_turn_count=remaining_turns,
            parent_turn_info=parent_turn_info,
            initial_main_history=initial_history
        )
        
        handoff_result = await handoff_manager.run(user_message=handoff_message["content"])
        logger.info("--- Handoff Workflow Finished ---")
        
        return handoff_result

    async def _handle_serial_handoff(self, policy: Dict[str, Any], state_args: Dict[str, Any], original_tool_name: str):
        target_agent_name = policy["target_agent"]
        # The handoff's system-level instruction is sourced from the policy configuration.
        # If not found, the target agent's own default instruction will be used.
        handoff_instruction = policy.get("handoff_instruction")
        interval_seconds = state_args.get("interval_seconds") or policy.get("interval_seconds", 0)
        handoff_context_source = policy.get("handoff_context_source", None)
        handoff_data = state_args.get("data", self.state.data)
        
        data_chunks = handoff_data.get("chunks", [handoff_data])
        aggregated_results = {}

        logger.info(f"Starting serial handoff to '{target_agent_name}' with {interval_seconds}s interval...")

        for i, chunk in enumerate(data_chunks):
            chunk_num = i + 1
            logger.debug(f"Processing chunk {chunk_num}/{len(data_chunks)}...")
            
            remaining_turns = self.max_turn_count - self.current_turn_count
            if remaining_turns < 1:
                logger.warning(f"Not enough turns left for serial handoff.")
                aggregated_results[f"chunk_{chunk_num}"] = "Handoff failed: Maximum turn count exceeded."
                break

            handoff_context = self.managers["builder"].build([target_agent_name])
            
            # Override the target agent's instructions if a handoff_instruction is provided
            if handoff_instruction:
                handoff_context.instructions = handoff_instruction

            parent_turn_info = f"Handoff from {self.context.agent_name} (Turn {self.current_turn_count}/{self.max_turn_count})"
            
            # --- Message Construction Logic ---
            handoff_message = None
            initial_history = None
            if handoff_context_source:
                if handoff_context_source == "turn_history":
                    initial_history = list(self.turn_history)
                handoff_message = self._build_handoff_message(handoff_context_source, chunk, chunk_num)
            else:
                instruction = state_args.get("instruction", "")
                data_str = self._format_data_to_string(chunk)
                combined_prompt = f"{instruction}, {data_str}" if instruction and data_str else instruction or data_str
                handoff_message = {"role": "user", "content": [{"type": "input_text", "text": combined_prompt}]}

            handoff_manager = PolicyManager(
                context=handoff_context, 
                managers=self.managers, 
                max_turn_count=remaining_turns,
                parent_turn_info=parent_turn_info,
                initial_main_history=initial_history
            )
            
            chunk_result = await handoff_manager.run(user_message=handoff_message["content"])
            aggregated_results[f"chunk_{chunk_num}"] = chunk_result
            
            if i < len(data_chunks) - 1 and interval_seconds > 0:
                await anyio.sleep(interval_seconds)
        
        logger.info("--- Serial Handoff Finished ---")
        return aggregated_results

    async def _handle_sequential_handoff(self, policy: Dict[str, Any], state_args: Dict[str, Any], original_tool_name: str):
        target_agents = policy["target_agents"]
        # The handoff's system-level instruction is sourced from the policy configuration.
        # If not found, the target agent's own default instruction will be used.
        handoff_instruction = policy.get("handoff_instruction")
        current_data = state_args.get("data", self.state.data)
        handoff_context_source = policy.get("handoff_context_source", None)

        logger.info(f"Starting sequential handoff to agents: {target_agents}")

        for i, agent_name in enumerate(target_agents):
            logger.info(f"--- Sequential Handoff: Step {i+1}/{len(target_agents)} -> {agent_name} ---")
            
            remaining_turns = self.max_turn_count - self.current_turn_count
            if remaining_turns < 1:
                logger.error("Handoff failed: Maximum turn count exceeded before completing the sequence.")
                current_data = {"status": "error", "data": "Handoff failed: Maximum turn count exceeded."}
                break

            handoff_context = self.managers["builder"].build([agent_name])
            instruction = handoff_instruction[i] if isinstance(handoff_instruction, list) else handoff_instruction

            # Override the target agent's instructions if a handoff_instruction is provided
            if instruction:
                handoff_context.instructions = instruction

            parent_turn_info = f"Handoff from {self.context.agent_name} (Turn {self.current_turn_count}/{self.max_turn_count})"
            
            # --- Message Construction Logic ---
            handoff_message = None
            initial_history = None
            if handoff_context_source:
                if handoff_context_source == "turn_history":
                    initial_history = list(self.turn_history)
                handoff_message = self._build_handoff_message(handoff_context_source, current_data)
            else:
                llm_instruction = state_args.get("instruction", "")
                data_str = self._format_data_to_string(current_data)
                combined_prompt = f"{llm_instruction}, {data_str}" if llm_instruction and data_str else llm_instruction or data_str
                handoff_message = {"role": "user", "content": [{"type": "input_text", "text": combined_prompt}]}


            handoff_manager = PolicyManager(
                context=handoff_context, 
                managers=self.managers, 
                max_turn_count=remaining_turns,
                parent_turn_info=parent_turn_info,
                initial_main_history=initial_history
            )
            
            current_data = await handoff_manager.run(user_message=handoff_message["content"])
            logger.info(f"--- Step {i+1} ({agent_name}) Finished ---")

        logger.info("--- Sequential Handoff Finished ---")
        return current_data

    async def _handle_parallel_handoff(self, policy: Dict[str, Any], state_args: Dict[str, Any], original_tool_name: str):
        target_agent_name = policy["target_agent"]
        # The handoff's system-level instruction is sourced from the policy configuration.
        # If not found, the target agent's own default instruction will be used.
        handoff_instruction = policy.get("handoff_instruction")
        handoff_context_source = policy.get("handoff_context_source", None)
        handoff_data = state_args.get("data", self.state.data)
        
        data_chunks = handoff_data.get("chunks", [handoff_data])
        logger.info(f"Starting parallel handoff to '{target_agent_name}' for {len(data_chunks)} chunks...")

        tasks = [
            self._run_single_handoff(target_agent_name, handoff_instruction, chunk, i + 1, handoff_context_source, state_args)
            for i, chunk in enumerate(data_chunks)
        ]
        results = await asyncio.gather(*tasks)
        aggregated_results = {f"chunk_{i+1}": result for i, result in enumerate(results)}

        logger.info("--- Parallel Handoff Finished ---")
        return aggregated_results

    async def _handle_handoff_feed_chunks(self, policy: Dict[str, Any], state_args: Dict[str, Any], original_tool_name: str):
        """
        Handles the 'handoff_feed_chunks' action by creating a sub-agent that
        pulls data chunks via a tool, rather than having them pushed.
        """
        # 1. Extract policy details and data
        target_agent_name = policy.get("target_agent")
        final_instruction = policy.get("final_instruction", "You have now received all data chunks. Please perform the original user's request based on the complete dataset you have accumulated.")
        handoff_data = state_args.get("data", self.state.data)

        if not (isinstance(handoff_data, dict) and "chunks" in handoff_data):
            logger.warning(f"Data for 'handoff_feed_chunks' is not chunked. Falling back to standard handoff.")
            return await self._handle_handoff(policy, state_args, original_tool_name)

        data_chunks = handoff_data.get("chunks", [])
        total_chunks = len(data_chunks)
        logger.info(f"Starting 'handoff_feed_chunks' to agent '{target_agent_name}' with {total_chunks} chunks.")

        # 2. Define the protocol's data-request tool
        request_chunk_tool_schema = {
            "type": "function",
            "function": {
                "name": "request_next_chunk",
                "description": "Call this function to request a specific chunk of the dataset by its number.",
                "parameters": {
                    "type": "object",
                    "properties": {"chunk_number": {"type": "integer", "description": "The number of the data chunk you want to receive (starting from 1)."}},
                    "required": ["chunk_number"]
                }
            }
        }
        protocol_tool = Tool(name="request_next_chunk", source_type="protocol", source_name="protocol", schema=request_chunk_tool_schema)

        # 3. Build the execution context for the disposable agent
        handoff_context = self.managers["builder"].build([target_agent_name])
        handoff_context.tools.append(protocol_tool)

        # Override instruction if provided in the policy
        handoff_instruction = policy.get("handoff_instruction")
        if handoff_instruction:
            handoff_context.instructions = handoff_instruction

        # 4. Create the disposable PolicyManager
        remaining_turns = self.max_turn_count - self.current_turn_count
        if remaining_turns < total_chunks + 2: # Estimate turns needed
            logger.error(f"Not enough turns left for handoff_feed_chunks. Required: ~{total_chunks + 2}, Available: {remaining_turns}")
            return {"status": "error", "data": "Handoff failed: Maximum turn count exceeded."}

        disposable_manager = PolicyManager(handoff_context, self.managers, max_turn_count=remaining_turns)
        disposable_manager.state = State(data=None, context={}) # Initialize state

        # 5. Define and prepend the new system prompt
        system_prompt = f"""You are a specialized data analysis sub-agent. Your purpose is to analyze a large dataset that has been split into {total_chunks} parts called 'chunks'.

**PROTOCOL INSTRUCTIONS:**
1. You must request each chunk of data sequentially by its number, starting with chunk 1.
2. To request a chunk, you MUST call the `request_next_chunk` tool with the correct `chunk_number`.
3. The tool will return the data for that chunk. When you have requested a chunk number that is out of range, the tool will inform you that all data has been sent.
4. Do NOT perform any final analysis until the tool tells you that you have received all data.
"""
        disposable_manager.context.instructions = system_prompt + "\n\n---\n\n" + disposable_manager.context.instructions

        # 6. The new "Ignition" prompt to kick off the process
        ignition_prompt = f"I have a dataset for you to analyze that is split into {total_chunks} chunks. Please begin by calling the `request_next_chunk` tool to retrieve chunk number 1."
        disposable_manager.turn_history.append({"role": "user", "content": [{"type": "input_text", "text": ignition_prompt}]})

        # 7. The new pull-based accumulation loop
        final_result = ""
        while True:
            # a. Call the LLM and get its response
            llm_response_messages, response_id = await disposable_manager._call_llm()
            if response_id:
                disposable_manager.state.context['llm_response_id'] = response_id
            disposable_manager.turn_history.extend(llm_response_messages)

            # b. Check for a premature final answer from a "smart" LLM
            final_assistant_message = next((m for m in llm_response_messages if m.get("role") == "assistant" and m.get("content")), None)
            if final_assistant_message:
                logger.info("Sub-agent provided a final answer prematurely. Accepting it as the result.")
                final_result = "".join(block.get("text", "") for block in final_assistant_message["content"] if block.get("type") == "output_text")
                break # Exit the loop as we have our final answer

            # c. If no final answer, expect a tool call
            fnc_call = next((msg for msg in llm_response_messages if msg.get("type") == "function_call"), None)

            # d. Validate the response against the protocol
            if not fnc_call or fnc_call.get("name") != "request_next_chunk":
                logger.error(f"Protocol violation: Agent did not call 'request_next_chunk' or provide a final answer. Response: {llm_response_messages}")
                return {"status": "error", "data": "Handoff failed due to protocol violation by the sub-agent."}

            try:
                args = json.loads(fnc_call.get("arguments", "{}"))
                requested_chunk_num = int(args.get("chunk_number"))
            except (json.JSONDecodeError, ValueError, TypeError) as e:
                logger.error(f"Protocol violation: Could not parse arguments from tool call: {fnc_call.get('arguments')}. Error: {e}")
                return {"status": "error", "data": "Handoff failed due to unparsable tool call arguments."}

            # e. Prepare the tool's response: either data or the termination signal
            tool_response_text = ""
            if 1 <= requested_chunk_num <= total_chunks:
                # Provide the requested chunk
                chunk_data = data_chunks[requested_chunk_num - 1]
                tool_response_text = json.dumps({
                    "status": "OK",
                    "chunk_number": requested_chunk_num,
                    "data": chunk_data
                })
                logger.info(f"Sub-agent requested chunk {requested_chunk_num}. Providing data.")
            else:
                # All chunks have been sent. Send the termination signal.
                logger.info(f"Sub-agent requested chunk {requested_chunk_num}, which is out of bounds. All chunks have been provided. Sending termination signal.")
                formatted_final_instruction = final_instruction.format(total_chunks=total_chunks)
                tool_response_text = json.dumps({
                    "status": "COMPLETE",
                    "message": formatted_final_instruction
                })

            tool_response_message = {
                "type": "function_call_output",
                "call_id": fnc_call.get("call_id"),
                "output": {"type": "text", "text": tool_response_text}
            }
            disposable_manager.turn_history.append(tool_response_message)

            # f. If the termination signal was sent, break the loop to get the final answer
            if "COMPLETE" in tool_response_text:
                break

        # 8. Get the final analysis from the sub-agent, but only if we don't already have it
        if not final_result:
            logger.info("Requesting final analysis from sub-agent.")
            final_llm_response_messages, _ = await disposable_manager._call_llm()
            
            final_assistant_message = next((m for m in final_llm_response_messages if m.get("role") == "assistant" and m.get("content")), None)
            if final_assistant_message:
                final_result = "".join(block.get("text", "") for block in final_assistant_message["content"] if block.get("type") == "output_text")
            else:
                final_result = "Error: Sub-agent did not provide a final response after receiving all data."

        logger.info(f"Handoff feed chunks finished. Final result: {final_result[:200]}...")
        return final_result

    async def _run_single_handoff(self, target_agent_name: str, instruction: str, chunk: Any, chunk_num: int, handoff_context_source: str, state_args: Dict[str, Any]) -> Any:
        logger.debug(f"Processing chunk {chunk_num} in parallel...")
        
        remaining_turns = self.max_turn_count - self.current_turn_count
        if remaining_turns < 1:
            return "Handoff failed: Maximum turn count exceeded."

        handoff_context = self.managers["builder"].build([target_agent_name])
        handoff_message = None
        initial_history = None
        if handoff_context_source:
            if handoff_context_source == "turn_history":
                initial_history = list(self.turn_history)
            handoff_message = self._build_handoff_message(handoff_context_source, chunk, chunk_num)
        else:
            llm_instruction = state_args.get("instruction", "")
            data_str = self._format_data_to_string(chunk)
            combined_prompt = f"{llm_instruction}, {data_str}" if llm_instruction and data_str else llm_instruction or data_str
            handoff_message = {"role": "user", "content": [{"type": "input_text", "text": combined_prompt}]}

        handoff_manager = PolicyManager(
            context=handoff_context, 
            managers=self.managers, 
            max_turn_count=remaining_turns,
            initial_main_history=initial_history
        )
        
        try:
            return await handoff_manager.run(user_message=handoff_message["content"])
        except Exception as e:
            log_exception(f"Parallel handoff for chunk {chunk_num}", e)
            return f"Error processing chunk {chunk_num}: {e}"

class AgentBuilder:
    def __init__(self, all_agent_configs, tool_manager: ToolManager):
        self.agents     = {a["name"]:a for a in all_agent_configs}
        self.tool_mgr   = tool_manager

    def build(self, agent_names: List[str]) -> ExecutionContext:
        base_config = self.agents[agent_names[0]]
        tools = self.tool_mgr.get_tools_for_agent(base_config)
        return ExecutionContext(
            agent_name=base_config.get("name", "unknown_agent"),
            provider=base_config.get("provider", "default"),
            api=base_config.get("api", "v1_chat_completions"),  # Default to v1_chat_completions
            model=base_config.get("model", "default"),
            model_settings=base_config.get("model_settings", {}),
            instructions=base_config.get("instruction", ""),
            tools=tools,
            policies=base_config.get("policies", {}),
            history_mode=base_config.get("history_mode", "full")
        )

# ------------------------------------------------- #
# Utillities Block
# ------------------------------------------------- #


import functools

def tool(name: str, description: str, tags: list = None):
    """A decorator to register a function as a tool.

    Args:
        name: The name of the tool.
        description: A description of what the tool does.
        tags: A list of tags for categorizing the tool.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        
        # Attach the metadata to the function object
        wrapper._is_tool = True
        wrapper._tool_name = name
        wrapper._tool_description = description
        wrapper._tool_tags = tags or []
        return wrapper
    return decorator


def log_event(event: str, data: dict = None):
    """
    Log a structured event (INFO level).
    """
    payload = {
        "event": event,
        "details": data or {},
        "uuid": str(uuid.uuid4()),
        "timestamp": datetime.now().isoformat()
    }
    logging.info(payload)

def log_exception(context: str, exc: Exception):
    """
    Log an exception with traceback (ERROR level).
    """
    logging.error(f"{context} -- {exc}", exc_info=True)

def estimate_tokens(messages: list, model_name: str = "gpt-4"):
    """
    Very rough token estimate based on tiktoken.
    """
    encoding_name = "cl100k_base"
    try:
        enc = tiktoken.get_encoding(encoding_name)
    except Exception:
        enc = tiktoken.get_encoding("cl100k_base")
    num_tokens = 0
    for msg in messages:
        num_tokens += 4
        for v in msg.values():
            num_tokens += len(enc.encode(str(v)))
    num_tokens += 2
    return num_tokens





