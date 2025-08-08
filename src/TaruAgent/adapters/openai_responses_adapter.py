import logging
import json
from openai import OpenAI
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

def _translate_history_to_input(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Translates the agent's internal history format (now aligned with /v1/responses input)
    into the structured 'input' array required by the /v1/responses endpoint,
    supporting multimodal content.
    """
    input_events = []
    for msg in messages:
        # If it's a message with a role (user, system, assistant text)
        if "role" in msg:
            translated_content_blocks = []
            for block in msg.get("content", []):
                if block.get("type") == "input_text":
                    translated_content_blocks.append({"type": "input_text", "text": block.get("text", "")})
                elif block.get("type") == "output_text":
                    translated_content_blocks.append({"type": "output_text", "text": block.get("text", "")})
                elif block.get("type") == "image_url":
                    image_url_obj = block.get("image_url", {})
                    if isinstance(image_url_obj, dict) and "url" in image_url_obj:
                        translated_content_blocks.append({
                            "type": "input_image",
                            "image_url": image_url_obj["url"]
                        })
                elif block.get("type") == "image_base64":
                    image_data = block.get("image_base64", {})
                    media_type = image_data.get("media_type", "image/jpeg")
                    b64_data = image_data.get("data")
                    if b64_data:
                        data_uri = f"data:{media_type};base64,{b64_data}"
                        translated_content_blocks.append({
                            "type": "input_image",
                            "image_url": data_uri
                        })
                # Add other content types as needed
            input_events.append({"role": msg["role"], "content": translated_content_blocks})
        # If it's a function call or function call output (from the new internal format)
        elif "type" in msg and msg["type"] == "function_call":
            input_events.append(msg)
        elif "type" in msg and msg["type"] == "function_call_output":
            # The /v1/responses API expects the 'output' field to be a string.
            # If it's an object, serialize it to a JSON string.
            output_data = msg.get("output", {})
            if isinstance(output_data, dict):
                output_string = json.dumps(output_data)
            else:
                output_string = str(output_data)

            input_events.append({
                "type": "function_call_output",
                "call_id": msg.get("call_id"),
                "output": output_string
            })
        else:
            logger.warning(f"Unknown message format in history: {msg}")
    return input_events

def _translate_api_response_to_internal_format(api_response) -> List[Dict[str, Any]]:
    """
    Translates the raw response object from the OpenAI client (responses.create)
    into the agent's new internal history format, which is aligned with the
    /v1/responses API's input/output structure.

    Returns:
        A list of message dictionaries suitable for direct appending to history.
    """
    internal_messages = []

    if hasattr(api_response, 'output') and api_response.output:
        for output_item in api_response.output:
            if hasattr(output_item, 'type'):
                if output_item.type == 'message':
                    # This is a message block (text, code, etc.)
                    content_blocks = []
                    if hasattr(output_item, 'content') and output_item.content:
                        for content_block in output_item.content:
                            if hasattr(content_block, 'type'):
                                block_type = content_block.type
                                if block_type == 'output_text' and hasattr(content_block, 'text'):
                                    content_blocks.append({"type": "output_text", "text": content_block.text})
                                elif block_type == 'code' and hasattr(content_block, 'code'):
                                    content_blocks.append({"type": "code", "code": content_block.code})
                                elif block_type == 'image_url' and hasattr(content_block, 'image_url'):
                                    content_blocks.append({"type": "image_url", "image_url": content_block.image_url})
                                elif block_type == 'image_base64' and hasattr(content_block, 'image_base64'):
                                    content_blocks.append({"type": "image_base64", "image_base64": content_block.image_base64})
                    # Assistant text/code response
                    internal_messages.append({"role": "assistant", "content": content_blocks})

                elif output_item.type == 'function_call':
                    # This is a tool call block
                    # Ensure arguments are a JSON string
                    arguments_str = output_item.arguments
                    if not isinstance(arguments_str, str):
                        arguments_str = json.dumps(arguments_str)

                    internal_messages.append({
                        "type": "function_call",
                        "call_id": output_item.call_id,
                        "name": output_item.name,
                        "arguments": arguments_str # Ensure this is a JSON string
                    })
    return internal_messages

def _translate_tools_for_responses_api(tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Translates the tools format from chat.completions style to responses.create style.
    (This function remains largely the same as it deals with tool definitions, not history)
    """
    translated_tools = []
    for tool in tools:
        if tool.schema.get("type") == "function" and "function" in tool.schema:
            func_details = tool.schema["function"]
            translated_tools.append({
                "type": "function",
                "name": func_details.get("name"),
                "description": func_details.get("description"),
                "parameters": func_details.get("parameters"),
            })
        else:
            translated_tools.append(tool)
    return translated_tools

async def call_responses_api(
    client: OpenAI,
    model: str,
    messages: List[Dict[str, Any]], # This will now be in the new internal format
    tools, # Removed type hint for diagnostic purposes
    instructions: str,
    model_settings: Dict[str, Any],
    previous_response_id: Optional[str] = None,
    history_mode: Optional[str] = None
) -> List[Dict[str, Any]]: # Returns a list of messages in the new internal format
    """
    The primary adapter function. It translates data, calls the API,
    and translates the response back into the agent's standard format.

    Returns:
        A list of message dictionaries in the new internal format.
    """
    logger.debug("--- OpenAI Responses Adapter: Starting ---")
    try:
        # Step 1: Translate the agent's history into the API's input format.
        if history_mode == "response_id":
            api_input = _translate_history_to_input([messages[-1]]) if messages else []
        else:
            api_input = _translate_history_to_input(messages)
        logger.debug(f"Translated API Input: {api_input}")

        # Step 2: Call the OpenAI API.
        translated_tools = _translate_tools_for_responses_api(tools)

        api_kwargs = model_settings.copy() if model_settings else {}
        if history_mode == "response_id" and previous_response_id:
            api_kwargs['previous_response_id'] = previous_response_id

        logger.debug("Calling client.responses.create with arguments:")
        logger.debug(f"  model: {model}")
        logger.debug(f"  input: {api_input}")
        logger.debug(f"  tools: {translated_tools}")
        logger.debug(f"  instructions: {instructions}")
        logger.debug(f"  model_settings: {api_kwargs}")

        raw_response = await client.responses.create(
            model=model,
            input=api_input,
            tools=translated_tools,
            instructions=instructions,
            **api_kwargs
        )
        logger.debug(f"Raw API Response: {raw_response}")

        # Step 3: Translate the raw API response back to our new internal format.
        internal_format_messages = _translate_api_response_to_internal_format(raw_response)
        logger.debug(f"Translated to New Internal Format: {internal_format_messages}")

        return internal_format_messages, raw_response.id

    except Exception as e:
        logger.error(f"Error in OpenAI Responses Adapter: {e}", exc_info=True)
        # Return an error message in the new internal format
        return [{"role": "assistant", "content": [{"type": "output_text", "text": f"API call failed: {e}"}]}], None
