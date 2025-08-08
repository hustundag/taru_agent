import logging
import uuid
from typing import List, Dict, Any, Optional, Union
import json

from google.genai import types

logger = logging.getLogger(__name__)


def _internal_to_gemini_contents(messages: List[Dict[str, Any]]) -> List[types.Content]:
    gemini_contents = []
    for msg in messages:
        if msg.get("type") == "function_call":
            arguments = msg.get("arguments", {})
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except (json.JSONDecodeError, TypeError) as e:
                    logger.error(f"Error decoding tool arguments JSON: {e}. Raw arguments: {arguments}")
                    arguments = {}  # Fallback to empty dict on error
                except Exception:
                    logger.warning("Could not parse function_call arguments string to dict, passing as empty dict.")
                    arguments = {}
            part = types.Part.from_function_call(
                name=msg["name"],
                args=arguments,
            )
            gemini_msg = types.ModelContent(
                parts=[part]
            )
            gemini_contents.append(gemini_msg)
        elif msg.get("type") == "function_call_output":
            tool_name = msg.get("name")
            tool_output = msg.get("output", "")

            # The Gemini API's FunctionResponse expects a dictionary payload.
            # This logic handles the various forms the tool_output can take.
            final_response_dict = {}
            if isinstance(tool_output, dict) and tool_output.get("type") == "text" and "text" in tool_output:
                # This is the specific format from TaruAgent, where the actual result is in the 'text' field.
                text_content = tool_output["text"]
                try:
                    # The content might be a JSON string (e.g., '{"a": 1}' or '4').
                    parsed_content = json.loads(text_content)
                    if isinstance(parsed_content, dict):
                        final_response_dict = parsed_content  # Use the dict directly.
                    else:
                        final_response_dict = {"result": parsed_content} # Wrap primitives/lists.
                except (json.JSONDecodeError, TypeError):
                    # It's just a plain string.
                    final_response_dict = {"result": text_content}
            elif isinstance(tool_output, dict):
                # It's a dictionary, but not the TaruAgent format. Assume it's the intended output.
                final_response_dict = tool_output
            else:
                # It's a primitive type (string, number, etc.). Wrap it.
                final_response_dict = {"result": tool_output}

            gemini_contents.append(
                types.Content(
                    role="tool",
                    parts=[types.Part.from_function_response(name=tool_name, response=final_response_dict)]
                )
            )
        else:
            role = msg.get("role", "user")
            if role == "assistant":
                role = "model"
            parts = []
            for block in msg.get("content", []):
                if block.get("type") in ("input_text", "output_text"):
                    parts.append(types.Part.from_text(text=block.get("text", "")))
                elif block.get("type") == "image_url":
                    url = block.get("image_url", {}).get("url", "")
                    parts.append(types.Part.from_data(data=url))
                elif block.get("type") == "image_base64":
                    image_data = block.get("image_base64", {})
                    b64_data = image_data.get("data", "")
                    parts.append(types.Part.from_data(data=b64_data))
            gemini_contents.append(types.Content(role=role, parts=parts))
    return gemini_contents



def _internal_to_gemini_tools(tools: List[Any]) -> Optional[List[types.Tool]]:
    gemini_tools = []
    for tool in tools:
        if isinstance(tool, types.Tool):
            gemini_tools.append(tool)
        elif hasattr(tool, "schema"):
            schema = tool.schema
            if schema.get("type") == "function" and "function" in schema:
                fn = schema["function"]
                gemini_tools.append(
                    types.Tool(
                        function_declarations=[
                            types.FunctionDeclaration(
                                name=fn.get("name"),
                                description=fn.get("description", ""),
                                parameters=fn.get("parameters", {}),
                            )
                        ]
                    )
                )
        elif isinstance(tool, dict):
            schema = tool.get("schema", {})
            if schema.get("type") == "function" and "function" in schema:
                fn = schema["function"]
                gemini_tools.append(
                    types.Tool(
                        function_declarations=[
                            types.FunctionDeclaration(
                                name=fn.get("name"),
                                description=fn.get("description", ""),
                                parameters=fn.get("parameters", {}),
                            )
                        ]
                    )
                )
        else:
            logger.warning(f"Skipping unsupported tool type: {type(tool)} -- {tool}")
    return gemini_tools if gemini_tools else None



def _gemini_response_to_internal(resp) -> List[Dict[str, Any]]:
    """
    Translates Gemini SDK response to internal message format.
    Handles function calls and assistant outputs.
    """
    result = []
    for cand in getattr(resp, 'candidates', []):
        for part in getattr(cand.content, 'parts', []):
            if hasattr(part, "function_call") and part.function_call:
                fc = part.function_call
                args = getattr(fc, "args", {})
                result.append({
                    "type": "function_call",
                    "call_id": str(uuid.uuid4()),
                    "name": getattr(fc, "name", None),
                    "arguments": json.dumps(args) if not isinstance(args, str) else args,
                })
            elif hasattr(part, "text"):
                result.append({
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": part.text}],
                })
    if hasattr(resp, 'text') and resp.text and not result:
        result.append({
            "role": "assistant",
            "content": [{"type": "output_text", "text": resp.text}],
        })
    return result



async def call_gemini_api(
    client,
    model: str,
    messages: List[Dict[str, Any]],
    tools: List[Dict[str, Any]],
    instructions: str,
    model_settings: Dict[str, Any],
    history_mode: Optional[str] = None
) -> List[Dict[str, Any]]:
    logger.debug("--- Gemini Adapter: Starting ---")
    try:
        gemini_contents = _internal_to_gemini_contents(messages)
        logger.debug(f"Gemini API input contents: {gemini_contents}")

        if instructions and not any(
            getattr(c, "role", "") == "user" and instructions in (p.text for p in getattr(c, "parts", []))
            for c in gemini_contents
        ):
            gemini_contents.insert(
                0, types.Content(role="user", parts=[types.Part.from_text(text=instructions)])
            )

        gemini_tools = _internal_to_gemini_tools(tools)
        logger.debug(f"Gemini tools for config: {gemini_tools}")

        import asyncio
        loop = asyncio.get_event_loop()

        def sync_generate_content():
            config_kwargs = dict(model_settings) if model_settings else {}
            
            if gemini_tools:
                config_kwargs['tools'] = gemini_tools
            else:
                # If there are no tools, ensure tool_config is not passed.
                config_kwargs.pop('tool_config', None)

            config = types.GenerateContentConfig(**config_kwargs) if config_kwargs else None

            logger.debug("Calling client.models.generate_content with arguments:")
            logger.debug(f"  model: {model}")
            logger.debug(f"  contents: {gemini_contents}")
            logger.debug(f"  config: {config}")

            return client.models.generate_content(
                model=model,
                contents=gemini_contents,
                config=config,
            )

        response = await loop.run_in_executor(None, sync_generate_content)

        logger.debug(f"Gemini API raw response: {response}")

        internal_format = _gemini_response_to_internal(response)
        logger.debug(f"Gemini internal format response: {internal_format}")

        return internal_format
    except Exception as e:
        logger.error(f"Gemini API call failed: {e}", exc_info=True)
        return [{"role": "assistant", "content": [{"type": "output_text", "text": f"API call failed: {e}"}]}]

