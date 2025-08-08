import logging
from openai import OpenAI
from typing import List, Dict, Any, Optional
import json
import uuid

logger = logging.getLogger(__name__)

previous_response_id: Optional[str] = None

def _translate_internal_to_chat_completions_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Translates the agent's new internal history format (aligned with /v1/responses)
    into the messages format required by the chat.completions API, supporting multimodal content.
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
                    # OpenAI expects base64 images as a data URI within image_url
                    image_data = block.get("image_base64", {})
                    media_type = image_data.get("media_type", "image/jpeg")  # Default to jpeg
                    b64_data = image_data.get("data")
                    if b64_data:
                        data_uri = f"data:{media_type};base64,{b64_data}"
                        content_parts.append({"type": "image_url", "image_url": {"url": data_uri}})
            chat_messages.append({"role": msg["role"], "content": content_parts})
        elif msg.get("role") == "assistant":
            # Assistant messages can have text content or tool_calls
            if msg.get("content"):
                content_text = "".join(block.get("text", "") for block in msg.get("content", []) if block.get("type") == "output_text")
                chat_messages.append({"role": "assistant", "content": content_text})
            elif msg.get("tool_calls"): # This path is for internal assistant messages that already have tool_calls
                chat_messages.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": msg["tool_calls"]
                })
            else:
                # Fallback for assistant messages without content or tool_calls
                chat_messages.append({"role": "assistant", "content": ""})
        elif msg.get("type") == "function_call":
            # This is an internal representation of an assistant tool call
            # It needs to be converted into the 'assistant' role with 'tool_calls'
            chat_messages.append({
                "role": "assistant",
                "content": None, # Content must be None when tool_calls are present
                "tool_calls": [
                    {
                        "id": msg["call_id"],
                        "type": "function",
                        "function": {
                            "name": msg["name"],
                            "arguments": msg["arguments"]
                        }
                    }
                ]
            })
        elif msg.get("type") == "function_call_output":
            # This is an internal representation of a tool output
            # It needs to be converted into the 'tool' role
            output_data = msg.get("output", {})
            # Ensure output_data is a string, as expected by the API
            output_text = output_data if isinstance(output_data, str) else output_data.get("text", "")
            
            chat_messages.append({
                "role": "tool",
                "tool_call_id": msg["call_id"],
                "content": output_text
            })
        else:
            logger.warning(f"Unknown message format in internal history for chat.completions: {msg}")
    return chat_messages

def _translate_internal_to_chat_completions_tools(tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Translates the agent's internal tool schema format into the tools format
    required by the chat.completions API.
    """
    chat_tools = []
    for tool in tools:
        if tool.schema.get("type") == "function" and "function" in tool.schema:
            chat_tools.append({
                "type": "function",
                "function": {
                    "name": tool.schema["function"].get("name"),
                    "description": tool.schema["function"].get("description"),
                    "parameters": tool.schema["function"].get("parameters"),
                }
            })
    return chat_tools

def _translate_chat_completions_response_to_internal(api_response) -> List[Dict[str, Any]]:
    """
    Translates the raw response object from chat.completions.create
    into the agent's new internal history format.
    """
    internal_messages = []
    message = api_response.choices[0].message.model_dump()

    if message.get('tool_calls'):
        for tc in message['tool_calls']:
            internal_messages.append({
                "type": "function_call",
                "call_id": tc['id'],
                "name": tc['function']['name'],
                "arguments": tc['function']['arguments']
            })
    else:
        # Standard text response
        content_text = message.get("content", "")
        internal_messages.append({"role": "assistant", "content": [{"type": "output_text", "text": content_text}]})

    return internal_messages

async def call_chat_completions_api(
    client: OpenAI,
    model: str,
    messages: List[Dict[str, Any]], # This will be in the new internal format
    tools: List[Dict[str, Any]],
    instructions: str,
    model_settings: Dict[str, Any],
    history_mode: Optional[str] = None
) -> List[Dict[str, Any]]: # Returns a list of messages in the new internal format
    """
    Adapter function for OpenAI chat.completions API.
    """
    logger.debug("--- OpenAI Chat Completions Adapter: Starting ---")
    try:
        # Step 1: Translate the agent's internal history into chat.completions format.
        api_messages = _translate_internal_to_chat_completions_messages(messages)
        logger.debug(f"Translated API Messages for chat.completions: {api_messages}")

        # Step 2: Translate tools.
        api_tools = _translate_internal_to_chat_completions_tools(tools)
        logger.debug(f"Translated API Tools for chat.completions: {api_tools}")

        # Add system instruction as a system message if not already present
        if instructions and not any(m.get("role") == "system" for m in api_messages):
            api_messages.insert(0, {"role": "system", "content": instructions})

        # Prepare keyword arguments for the API call, ensuring tool_choice is handled correctly.
        api_kwargs = model_settings.copy() if model_settings else {}
        if api_tools:
            if 'tool_choice' not in api_kwargs:
                api_kwargs['tool_choice'] = "auto"
        else:
            # If there are no tools, there must be no tool_choice.
            api_kwargs.pop('tool_choice', None)

        logger.debug("Calling client.chat.completions.create with arguments:")
        logger.debug(f"  model: {model}")
        logger.debug(f"  messages: {api_messages}")
        logger.debug(f"  tools: {api_tools}")
        logger.debug(f"  model_settings (with final tool_choice): {api_kwargs}")

        raw_response = await client.chat.completions.create(
            model=model,
            messages=api_messages,
            tools=api_tools if api_tools else None, # Pass None if no tools
            **(api_kwargs) # Pass the processed model settings
        )
        logger.debug(f"Raw API Response from chat.completions: {raw_response}")

        # Step 3: Translate the raw API response back to our new internal format.
        internal_format_messages = _translate_chat_completions_response_to_internal(raw_response)
        logger.debug(f"Translated to New Internal Format from chat.completions: {internal_format_messages}")

        return internal_format_messages, raw_response.id

    except Exception as e:
        logger.error(f"Error in OpenAI Chat Completions Adapter: {e}", exc_info=True)
        return [{"role": "assistant", "content": [{"type": "output_text", "text": f"API call failed: {e}"}]}], None