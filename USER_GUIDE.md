# TaruAgent Framework

**TaruAgent is a flexible, state-driven framework for building sophisticated, multi-agent enterprise applications. It is designed to address privacy and security concerns in AI adoption while managing the limited context size of Large Language Models (LLMs). You can use TaruAgent with leading providers like OpenAI and Google Gemini, as well as with your own local deployments via Ollama.**

Whether you're a seasoned AI developer or just starting, TaruAgent provides the tools you need to bring your AI agent ideas to life with minimal effort.

## ✨ Capabilities

The framework has the following set of features:

*   **🤖 Multi-Agent Orchestration:** Go beyond single-agent systems to organize and orchestrate model communication with your existing code.
*   **🤝 Advanced Handoff Strategies:** Natively supports multiple handoff flavors, including `serial`, `parallel`, and `sequential` pipelines, allowing agents to delegate tasks and collaborate effectively.
*   **🔌 Multi-Provider & Multi-Model Support:** Seamlessly switch between providers like **OpenAI**, **Google** APIs, or local deployments with **Ollama**. The next version will include support for any OpenAI API-compatible local vLLM deployment.
*   **🔧 Local and MCP Tools:** Integrate any of your Python code as a local tool using a simple decorator. Supports remote tools utilizing the Model Context Protocol (MCP).
*   **📜 Policy-Driven State Machine:** A transparent state machine architecture allows you to implement complex logic, loops, and guardrails with a simple YAML configuration file.
*   **🚀 Simplified Setup with `TaruRunner`:** A high-level facade that simplifies initialization and execution, getting you up and running faster.
*   **📝 Rich History Management:** Provides multiple history modes (`full`, `clean`, `none`, `response_id`) to control context and optimize for different use cases.
*   **🔍 Extensive Logging and Error Handling:** Comprehensive logging throughout the code and robust error handling to help you debug application issues during development.

## 🚀 Quick Start

Get your first TaruAgent running in just a few minutes.

### 1. Prerequisites

*   Python 3.12+
*   `pip3` for package installation
*   An environment variable for your chosen LLM provider's API key (e.g., `OPENAI_API_KEY`).

### 2. Installation Options

You can get started with TaruAgent in one of three ways:

#### a) Pip Install from Source (Recommended for Development)

This method installs the framework in editable mode, so any changes you make to the source are immediately available.

```bash
# Clone the repository (if you haven't already)
# git clone ...

# Navigate to the root of the project
cd /path/to/your/project

# Install the TaruAgent framework from the src directory
pip3 install -e src/

# Install other dependencies from the app's requirements file
pip3 install -r requirements.txt
```

#### b) Use the Pre-built Docker Image

The quickest way to get started. The Docker image comes with the framework and all necessary dependencies pre-installed.
It also includes simple AI CLI app example to get you started quickly. 
```bash
# Pull the latest Docker image
docker pull hustundag/taruagent:latest

# Run the container interactively
docker run -dit  \
  -e OPENAI_API_KEY="your_api_key" \
  -v /path/to/data:/apps/data \
  hustundag/taruagent:latest
```
Note on Configuration: The pre-built image is minimal and does not include a text editor. The recommended way to use your own config.yaml is to mount it directly from your computer using the -v flag for config.yaml. This
  lets you edit the file locally. 

#### c) Build Your Own Docker Image

Customize the environment by building your own Docker image. This is useful if you need to add extra packages.

```bash
# Build the image from the Dockerfile in the root directory
docker build -t my-custom-taruagent .

# Run your custom container
docker run -dit  \
  -e OPENAI_API_KEY="your_api_key" \
  -v /path/to/data:/apps/data \
  my-custom-taruagent
```

### 3. Configuration (`config.yaml`)

Create a `config.yaml` file. This minimal configuration defines an OpenAI provider and a simple agent.

```yaml
# config.yaml
providers:
  openai:
    api_key: "env:OPENAI_API_KEY"

agents:
  - name: "my_first_agent"
    provider: "openai"
    model: "gpt-4.1-mini"
    instruction: "You are a helpful assistant."
```

### 4. Create your Agent Runner

Create a Python file named `run_agent.py`:

```python
# run_agent.py

# run_agent.py
import asyncio
from TaruAgent import TaruRunner

async def main():
    async with TaruRunner(config_path="config.yaml") as runner:
        print("Your agent is ready! Type 'exit' to end the conversation.")
        while True:
            user_input = input("You: ")
            if user_input.lower() == 'exit':
                break
            message = [{'type': 'input_text', 'text': user_input}]
            response = await runner.run(agent_name="my_first_agent", user_message=message)
            print(f"Agent: {response}")

if __name__ == "__main__":
    asyncio.run(main())
```

### 5. Run it!

First, set your OpenAI API key as an environment variable:

```bash
export OPENAI_API_KEY="your_openai_api_key"
```

Now, run your agent:

```bash
python3 run_agent.py
```

## 🧠 Architecture and Configuration

This section provides a detailed breakdown of the framework's architecture, configuration, and advanced capabilities.

### 1. Architecture Overview

The framework is built around a central `PolicyManager` that controls an agent's lifecycle using a state machine. It interacts with several other managers:

*   **`TaruRunner`**: The main entry point and facade. It loads the configuration and initializes all other managers. It also maintains the persistent conversation history for each agent.
*   **`LLMClientManager`**: Manages API clients for different providers (OpenAI, Google, Ollama).
*   **`ToolManager`**: Discovers local tools, registers remote MCP tools, and dynamically generates "handoff tools".
*   **`MCPManager`**: Manages connections to remote tool servers using the Model Context Protocol (MCP).
*   **`CustomActionManager`**: Executes custom Python functions defined in policies.
*   **`AgentBuilder`**: Constructs the `ExecutionContext` for an agent, which is a dataclass containing everything the agent needs to run (model, tools, policies, etc.).

### 2. The State Machine (`PolicyManager`)

The core of the framework is a state machine that processes an agent's turn. Understanding these states is crucial for using policies effectively.

| State (`WorkflowState`)   | Trigger                               | Default Action           | `State.data` Contains                               |
| :------------------------ | :------------------------------------ | :----------------------- | :-------------------------------------------------- |
| `AT_USER_MESSAGE`         | New user input.                       | `return_to_llm`          | The user's message (in internal format).            |
| `AT_RETURN_TO_LLM`        | Transition action.                    | `call_llm`               | The data to be sent to the LLM.                     |
| `WAITING_LLM_RESPONSE`    | After LLM call.                       | `process_llm_response`   | The raw response from the LLM adapter.              |
| `AT_FUNCTION_CALL`        | LLM requests a tool.                  | `call_function`          | The arguments for the tool (as a JSON string).      |
| `AT_FUNCTION_CALL_RETURN` | Tool execution finishes.              | `return_to_llm`          | The result from the tool.                           |
| `AT_LLM_FINAL_RESPONSE`   | LLM gives a text answer.              | `return_to_user`         | The final text response.                            |
| `AT_USER_RETURN`          | Terminal state for the turn.          | `return_user_message`    | The final payload for the user.                     |

### 3. Configuration Deep Dive (`config.yaml`)

This is the reference for all configuration options. You can check the `testing` directory for different configurations.

```yaml
# 1. Provider Definitions
providers:
  openai:
    api_key: "env:OPENAI_API_KEY" # Use environment variable
  google:
    api_key: "env:GOOGLE_API_KEY"
  ollama:
    - name: "my_local_ollama_1"
      base_url: "http://localhost:11434"
    - name: "my_local_ollama_2"
      base_url: "http://10.21.2.101:11434"

# 2. Remote Tool Servers (MCP)
mcp_servers:
  - name: "unique_server_name"
    url: "http://host:port/sse"
    description: "Description of the server."

# 3. Local Tool Discovery
local_tools:
  - name: "tool_group_name" # Used to assign tools to agents
    path: "./path/to/tool_files/"

# 4. Agent Definitions
agents:
  - name: "my_agent"
    provider: "openai" 
    model: "gpt-4.1-mini" 
    api: "v1_responses" 
    instruction: "System prompt for the agent."
    max_turn_count: 50 # Max steps in the state machine per turn (each user message triggers around 7-15 turns depends on number of function calls)
    history_mode: "full" # "full", "clean", "none", "response_id"

    # Assign tools to this agent
    resources:
      local_tools:
        - group_name: "tool_group_name"
      mcp_tools:
        - server_name: "unique_server_name"
          filter:
            # permit: [ "tool_1", "tool_2" ] # Whitelist
            deny: [ "tool_3" ] # Blacklist

    # Expose handoffs as callable tools to the LLM
    handoff_as_tool_list:
      - action: "handoff" # The handoff type
        target_agent: "another_agent"
        handoff_description: "Description for the LLM."

    # Define custom behavior with policies
    policies:
      # See Policy Engine section for details
      at_function_call_return:
        - tool_name: "get_user_info"
          action_list:
            - action: "execute_custom_function"
              target: "my_module.actions.process_user_info"
```

**History Modes:**

*   `full`: Every message (user, assistant, tool call, tool response) is kept in the agent's history for the next turn.
*   `clean`: Only the initial user message and the final assistant response from a turn are kept.
*   `none`: History is not preserved between turns.
*   `response_id`: Supported only with OpenAI's `v1_responses` API. The framework does not track the actual history but passes the `response_id` from the previous turn, allowing OpenAI to maintain the session context. Note: Avoid using jump actions that bypass the LLM call when using this mode.

### 4. The Policy Engine

Policies are the core of TaruAgent's power. They allow you to intercept the state machine at any configurable state and execute a list of custom actions.

**Policy Structure:**

```yaml
policies:
  <state_name>: # e.g., at_function_call_return
    - tool_name: <tool_name> # Optional: apply only for a specific tool
      action_list:
        - action: <action_name>
          # ... action-specific parameters
```

**Policy Actions:**

*   **`execute_custom_function`**: Calls a Python function.
    *   `target`: The full import path to the function (e.g., `my_module.my_actions.my_function`).
    *   The function receives the current `State.data` and must return an `ActionOutput` object, which can be used to chain actions or modify data.
*   **State Jumps**: Immediately change the state of the machine. Useful for creating loops or short-circuiting the flow.
    *   `jump_to_user_message`
    *   `jump_to_return_to_llm`
    *   `jump_to_function_call_return`
    *   `jump_to_llm_final_response`
    *   `jump_to_user_return`
*   **Handoffs**: See the Handoff Strategies section.

### 5. Handoff Strategies

Handoffs allow agents to delegate tasks. They can be triggered from policies or exposed as tools for the LLM to call directly.

**Handoff Flavors (the `action` key):**

*   `handoff`: A simple, one-to-one handoff to another agent.
*   `handoff_serial`: Sends a list of data chunks to a single agent, one at a time, waiting for each to complete.
*   `handoff_parallel`: Sends a list of data chunks to a single agent, processing them all concurrently.
*   `handoff_sequential`: A multi-agent pipeline. The output of the first agent becomes the input for the second, and so on.
*   `handoff_feed_chunks`: A specialized protocol for feeding very large, chunked datasets to a disposable agent, protecting the main agent's context window. The sub-agent "pulls" data by calling a tool, rather than having it "pushed."

**Handoff Parameters:**

*   `target_agent`: The name of the agent to hand off to.
*   `target_agents`: A list of agent names for `handoff_sequential`.
*   `handoff_instruction`: A temporary system prompt for the target agent.
*   `handoff_context_source`: How to build the initial message for the target agent:
    *   `last_data_only` (default): The message is just the data from the previous step.
    *   `user_message_only`: The message is the original user input that started the turn.
    *   `user_message_and_data`: Combines the original user input and the last data.
    *   `turn_history`: The entire history of the current turn is passed to the target agent.

