import logging
import uuid
import json
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

async def call_ollama_api(
    client: Any, # Ollama client
    model: str,
    messages: List[Dict[str, Any]], # This will be in the new internal format
    tools: List[Dict[str, Any]],
    instructions: str,
    model_settings: Dict[str, Any],
    history_mode: Optional[str] = None
) -> List[Dict[str, Any]]: # Returns a list of messages in the new internal format
    """
    Adapter function for Ollama API.
    """
    logger.debug("--- Ollama Adapter: Starting ---")
    try:
        # Translate internal history to the format expected by Ollama (same as chat.completions)
        api_messages = _translate_internal_to_chat_completions_messages(messages)
        
        # Translate tools to the format expected by Ollama
        api_tools = _translate_internal_to_chat_completions_tools(tools)

        # Add system instruction if not already present
        if instructions and not any(m.get("role") == "system" for m in api_messages):
            api_messages.insert(0, {"role": "system", "content": instructions})

        logger.debug("Calling Ollama client.chat with arguments:")
        logger.debug(f"  model: {model}")
        logger.debug(f"  messages: {api_messages}")
        logger.debug(f"  tools: {api_tools}")
        logger.debug(f"  model_settings: {model_settings}")

        raw_response = await client.chat(
            model=model,
            messages=api_messages,
            tools=api_tools if api_tools else None,
            **model_settings
        )
        logger.debug(f"Raw API Response from Ollama: {raw_response}")

        # Convert Ollama response back to our new internal format
        internal_messages = []
        message = raw_response.get('message', {})

        if message.get('tool_calls'):
            for tc in message['tool_calls']:
                arguments = tc.get('function', {}).get('arguments', {})
                # Check for and unpack nested 'kwargs'
                if 'kwargs' in arguments and isinstance(arguments['kwargs'], dict):
                    arguments = arguments['kwargs']

                internal_messages.append({
                    "type": "function_call",
                    "call_id": tc.get('id', str(uuid.uuid4())), # Add unique ID if missing
                    "name": tc.get('function', {}).get('name'),
                    "arguments": json.dumps(arguments)
                })
        else:
            content_text = message.get("content", "")
            internal_messages.append({"role": "assistant", "content": [{"type": "output_text", "text": content_text}]})

        return internal_messages

    except Exception as e:
        logger.error(f"Error in Ollama Adapter: {e}", exc_info=True)
        return [{"role": "assistant", "content": [{"type": "output_text", "text": f"API call failed: {e}"}]}]

# Helper functions (mirrored from openai_chat_completions_adapter.py)

def _translate_internal_to_chat_completions_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Translates the agent's new internal history format into the messages format
    required by the chat.completions API, supporting multimodal content.
    """
    chat_messages = []
    for msg in messages:
        if msg.get("role") in ["user", "system"]:
            content_parts = []
            for block in msg.get("content", []):
                if block.get("type") == "input_text":
                    content_parts.append({"type": "text", "text": block.get("text", "")})
                elif block.get("type") == "image_url":
                    content_parts.append({"type": "image_url", "image_url": block.get("image_url", {})})
                elif block.get("type") == "image_base64":
                    # Ollama expects base64 images directly in the content list
                    content_parts.append({"type": "image_url", "image_url": {"url": f"data:{block.get("image_base64", {}).get("media_type")};base64,{block.get("image_base64", {}).get("data")}"}})
            # For Ollama, if content_parts is a list of text blocks, concatenate them into a single string
            # If it contains image_url, Ollama expects a list of dicts.
            # This is a simplification; a more robust solution might involve checking for image_url types.
            if all(block.get("type") == "text" for block in content_parts):
                chat_messages.append({"role": msg["role"], "content": " ".join(block.get("text", "") for block in content_parts)})
            else:
                # If there are other types (like image_url), keep it as a list of dicts
                chat_messages.append({"role": msg["role"], "content": content_parts})
        elif msg.get("role") == "assistant":
            if msg.get("content"):
                content_text = "".join(block.get("text", "") for block in msg.get("content", []) if block.get("type") == "output_text")
                chat_messages.append({"role": "assistant", "content": content_text})
            elif msg.get("tool_calls"):
                chat_messages.append({"role": "assistant", "content": None, "tool_calls": msg["tool_calls"]})
        elif msg.get("type") == "function_call":
                # Ensure arguments are a dictionary for Ollama client
                arguments_dict = {}
                try:
                    arguments_dict = json.loads(msg["arguments"])
                except (json.JSONDecodeError, TypeError) as e:
                    logger.warning(f"Could not decode function call arguments: {msg["arguments"]}. Error: {e}")

                chat_messages.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": msg["call_id"],
                        "type": "function",
                        "function": {"name": msg["name"], "arguments": arguments_dict}
                    }]
                })
        elif msg.get("type") == "function_call_output":
            output_data = msg.get("output", {})
            output_text = output_data if isinstance(output_data, str) else output_data.get("text", "")
            chat_messages.append({
                "role": "tool",
                "tool_call_id": msg["call_id"],
                "content": output_text
            })
    return chat_messages

def _translate_internal_to_chat_completions_tools(tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Translates the agent's internal tool schema into the tools format
    required by the chat.completions API.
    """
    chat_tools = []
    for tool in tools:
        if tool.schema.get("type") == "function" and "function" in tool.schema:
            chat_tools.append({"type": "function", "function": tool.schema["function"]})
    return chat_tools
