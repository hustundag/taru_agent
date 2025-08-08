#!/usr/bin/env python3
import asyncio
import sys
import logging
import os
from datetime import datetime
import json
from contextlib import AsyncExitStack
from typing import List, Optional, Dict, Any, Tuple
import uuid
import mimetypes
import base64
from TaruAgent import (
    TaruRunner,
    log_exception,
    LLMClientManager,
    estimate_tokens
    )

# --- Logging Setup ---
# This application configures the root logger, which will capture logs
# from the TaruAgent library as well.

class SensitiveRedactingFormatter(logging.Formatter):
    REDACT_KEYS = [
        "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY", "GEMINI_API_KEY"
    ]

    def __init__(self, fmt=None, datefmt=None, style='%', validate=True, *, defaults=None, session_id=None):
        super().__init__(fmt, datefmt, style, validate, defaults=defaults)
        self.session_id = session_id

    def redact(self, msg):
        for key in self.REDACT_KEYS:
            val = os.environ.get(key)
            if val and val in msg:
                msg = msg.replace(val, f"[{key}_REDACTED]")
        return msg

    def format(self, record):
        # Add session_id to the record so it can be used in the format string
        record.session_id = self.session_id
        formatted_msg = super().format(record)
        return self.redact(formatted_msg)

class JsonLineFormatter(logging.Formatter):
    REDACT_KEYS = [
        "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY", "GEMINI_API_KEY"
    ]
    
    def __init__(self, fmt=None, datefmt=None, style='%', validate=True, *, defaults=None, session_id=None):
        super().__init__(fmt, datefmt, style, validate, defaults=defaults)
        self.session_id = session_id

    def redact(self, msg):
        for key in self.REDACT_KEYS:
            val = os.environ.get(key)
            if val and val in msg:
                msg = msg.replace(val, f"[{key}_REDACTED]")
        return msg

    def format(self, record):
        original_message = record.getMessage()
        redacted_message = self.redact(original_message)
        out = {
            "time": self.formatTime(record, self.datefmt),
            "session_id": self.session_id,
            "level": record.levelname,
            "name": record.name,
            "msg": redacted_message
        }
        return json.dumps(out)

def setup_logging(session_id: str, log_config: Dict[str, Any]):
    """
    Configure the root logger for the application based on config.
    This will capture logs from both the application and the TaruAgent library.
    """
    root = logging.getLogger()
    if root.handlers:
        root.handlers.clear()

    # Get log level from config, default to INFO
    log_level_str = log_config.get('level', 'INFO').upper()
    level = getattr(logging, log_level_str, logging.INFO)
    root.setLevel(level)

    # Get log format(s) from config, default to text
    log_formats = log_config.get('log_format', 'text')
    if isinstance(log_formats, str):
        log_formats = [log_formats] # Ensure it's a list

    for format_type in log_formats:
        format_type = format_type.lower()
        if format_type == 'json':
            # JSONL file handler
            jh = logging.handlers.RotatingFileHandler(
                "aicli_debug.jsonl", maxBytes=5_000_000, backupCount=3, encoding="utf-8"
            )
            jh.setLevel(level)
            jh.setFormatter(JsonLineFormatter(session_id=session_id))
            root.addHandler(jh)
        elif format_type == 'text':
            # Rotating debug file handler
            fh = logging.handlers.RotatingFileHandler(
                "aicli_debug.log", maxBytes=5_000_000, backupCount=3, encoding="utf-8"
            )
            fh.setLevel(level)
            fh.setFormatter(SensitiveRedactingFormatter(
                "%(asctime)s - %(session_id)s - %(name)s - %(levelname)s - %(message)s",
                session_id=session_id
            ))
            root.addHandler(fh)


# --- End of Logging Setup ---
# Initialize a logger for the application
logger = logging.getLogger(__name__)

CHATLOG_DIR = "chat_logs"

def generate_filename(prefix, provider, model, ext, chat_log_dir):
    os.makedirs(chat_log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    safe_prefix = prefix.replace(" ", "_")
    safe_model = model.replace("/", "-").replace(":", "-")
    return f"{chat_log_dir}/{safe_prefix}_{provider}_{safe_model}_{timestamp}.{ext}"

def save_chat_log_json(messages, prefix, provider, model, chat_log_dir):
    filename = generate_filename(prefix, provider, model, "json", chat_log_dir)
    with open(filename, "w") as f:
        json.dump(messages, f, indent=2)
    return filename

def save_chat_log_txt(messages, prefix, provider, model, chat_log_dir):
    filename = generate_filename(prefix, provider, model, "txt", chat_log_dir)
    with open(filename, "w") as f:
        for msg in messages:
            role = msg.get('role', 'unknown').capitalize()
            content = ""
            if isinstance(msg.get('content'), list):
                for block in msg['content']:
                    if block.get('type') == 'input_text' or block.get('type') == 'output_text':
                        content += block.get('text', '')
                    elif block.get('type') == 'image_url':
                        content += f"<Image: {block.get('image_url', {}).get('url', 'unknown')}>"
                    elif block.get('type') == 'image_base64':
                        content += f"<Image: base64 data, type: {block.get('image_base64', {}).get('media_type', 'unknown')}>"
            else:
                content = str(msg.get('content', '')).strip()
            if not content and 'function_call' in msg:
                content = f"Function Call: {msg['function_call']}"
            f.write(f"{role}: {content}\n\n")
    return filename


def get_image_base64(image_path: str) -> Optional[Dict[str, str]]:
    """
    Reads an image file, encodes it to base64, and determines its media type.
    Returns a dictionary with 'data' (base64 string) and 'media_type', or None if an error occurs.
    """
    if not os.path.exists(image_path):
        logger.error(f"Image file not found: {image_path}")
        return None
    
    mime_type, _ = mimetypes.guess_type(image_path)
    if not mime_type or not mime_type.startswith("image/"):
        logger.error(f"Unsupported file type for image: {mime_type or 'unknown'}")
        return None

    try:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
        return {"data": encoded_string, "media_type": mime_type}
    except Exception as e:
        logger.error(f"Error encoding image {image_path}: {e}")
        return None

def multiline_input(prompt="User (finish with /send, /image <path>, /image_url <url>, or /exit on first line)> "):
    """
    Collects multiline user input until a sentinel is entered.
    Recognizes /image <path> for local image input and /image_url <url> for web image input.
    Returns (full_input_text, local_image_path, image_url, should_exit).
    """
    print(prompt)
    lines = []
    local_image_path = None
    image_url = None
    first_line = True
    while True:
        try:
            line = input()
        except EOFError:
            return None, None, None, True
        
        cmd = line.strip()
        if first_line and cmd.lower() in ("/exit", ":exit", "exit", "/quit", ":quit", "quit"):
            return None, None, None, True
        
        if cmd.lower().startswith("/image "):
            local_image_path = cmd[len("/image "):].strip()
            logger.info(f"Detected local image command: {local_image_path}")
            continue
        elif cmd.lower().startswith("/image_url "):
            image_url = cmd[len("/image_url "):].strip()
            logger.info(f"Detected image URL command: {image_url}")
            continue

        first_line = False
        if cmd.lower() == "/send":
            return '\n'.join(lines), local_image_path, image_url, False
        lines.append(line)



# ------------------------------------------------- #
# Main Execution Block
# ------------------------------------------------- #


def prompt_select(options: List[str], prompt: str, default: Optional[str] = None) -> str:
    """
    Presents a numbered list of options, prompts the user, and returns the selected string.
    Typing 'exit' or 'quit' at the prompt will terminate the program.
    """
    while True:
        for idx, opt in enumerate(options, start=1):
            print(f"{idx}) {opt}")
        suffix = f" [{default}]" if default else ""
        choice = input(f"{prompt}{suffix}: ").strip()

        # global exit
        if choice.lower() in ("exit", "quit"):
            print("Goodbye.")
            sys.exit(0)

        # default on empty
        if not choice and default:
            return default

        # numeric selection
        if choice.isdigit():
            n = int(choice) - 1
            if 0 <= n < len(options):
                return options[n]

        # exact match
        if choice in options:
            return choice

        print("Invalid selection, please try again.\n")

def select_and_build_context(
    cfg: Dict[str, Any],
) -> Tuple[str, str, str, str]:
    """
    Selects provider, model, agent, and API, returning the chosen names.
    """
    runflow = cfg.get("runflow", {})
    defaults = cfg.get("defaults", {})

    # --- 1. Flatten providers and Select Provider ---
    all_providers = []
    for name, config in cfg["providers"].items():
        if isinstance(config, list):
            for item in config:
                all_providers.append(item)
        else:
            config['name'] = name
            all_providers.append(config)

    provider_names = [p['name'] for p in all_providers]
    sel_prov_name = None

    if runflow.get("use_default_provider"):
        sel_prov_name = defaults.get("provider")
        if sel_prov_name not in provider_names:
            logger.warning(f"Default provider '{sel_prov_name}' not found. Falling back to prompt.")
            sel_prov_name = None
    if not sel_prov_name:
        sel_prov_name = prompt_select(provider_names, "Select provider (or 'exit')", provider_names[0] if provider_names else None)

    # --- 2. Select Model ---
    selected_provider_config = next((p for p in all_providers if p['name'] == sel_prov_name), None)
    if not selected_provider_config:
        logger.error(f"Could not find config for provider '{sel_prov_name}'")
        sys.exit(1)

    model_names = [str(m) for m in selected_provider_config.get("models", [])]
    sel_mod = None
    if runflow.get("use_default_model"):
        sel_mod = defaults.get("model")
        if sel_mod not in model_names:
            logger.warning(f"Default model '{sel_mod}' not found for provider. Falling back to prompt.")
            sel_mod = None
    if not sel_mod:
        sel_mod = prompt_select(model_names, f"Select model for '{sel_prov_name}'", model_names[0] if model_names else None)

    # --- 3. Select API ---
    api_types = ['v1_chat_completions', 'v1_responses', 'gemini_v1beta', 'ollama_openai_compatible']
    sel_api = prompt_select(api_types, "Select API type", api_types[0])

    # --- 4. Select Agent ---
    agent_names = [a["name"] for a in cfg.get("agents", [])]
    if not agent_names:
        logger.error("No agents defined in the configuration file.")
        sys.exit(1)
        
    sel_agent = None
    if runflow.get("use_default_agent"):
        sel_agent = defaults.get("agent")
        if sel_agent not in agent_names:
            logger.warning(f"Default agent '{sel_agent}' not found. Falling back to prompt.")
            sel_agent = None
    if not sel_agent:
        default_agent = agent_names[0] if agent_names else None
        sel_agent = prompt_select(agent_names, "Select agent", default_agent)

    return sel_prov_name, sel_mod, sel_agent, sel_api

async def main():
    # 1. Generate a unique ID for this session
    session_id = str(uuid.uuid4())

    # Load configuration first to get logging settings
    try:
        runner = TaruRunner(config_path="config.yaml")
        cfg = runner.config
        log_config = cfg.get('system', {}).get('logging', {})
        chat_log_dir = log_config.get('chat_log_dir', './chat_logs') # Default value
    except FileNotFoundError as e:
        print(f"Configuration file error: {e}")
        setup_logging(session_id=session_id, log_config={})
        logging.error(f"Configuration file error: {e}")
        return
    except Exception as e:
        setup_logging(session_id=session_id, log_config={})
        logging.critical(f"A critical error occurred during initial configuration: {e}", exc_info=True)
        return

    # 2. Set up logging with the loaded configuration
    setup_logging(session_id=session_id, log_config=log_config)
    logging.info("Taru Agent Framework Initialized.")

    try:
        runflow_config = cfg.get("runflow", {})
        if runflow_config.get("use_default_agent"):
            # Zero-prompt mode
            sel_agent = cfg.get("defaults", {}).get("agent")
            if not sel_agent:
                logging.error("'use_default_agent' is true, but no default agent is specified in the 'defaults' section.")
                return
            
            agent_config = next((a for a in cfg.get('agents', []) if a['name'] == sel_agent), None)
            if not agent_config:
                logging.error(f"Default agent '{sel_agent}' not found in agent definitions.")
                return

            sel_prov = agent_config.get('provider')
            sel_mod = agent_config.get('model')
            sel_api = agent_config.get('api', 'v1_chat_completions') # Fallback to default if not specified
            logging.info(f"Running in zero-prompt mode with default agent: {sel_agent}")

        else:
            # Interactive mode
            sel_prov, sel_mod, sel_agent, sel_api = select_and_build_context(cfg)
            logging.info(f"Context selected: Provider='{sel_prov}', Model='{sel_mod}', Agent='{sel_agent}', API='{sel_api}'")

            # Override the agent's config with the user's selection.
            agent_config_to_modify = next((a for a in runner.config['agents'] if a['name'] == sel_agent), None)
            if agent_config_to_modify:
                logging.debug(f"Overriding agent '{sel_agent}' config with user selection: Provider='{sel_prov}', Model='{sel_mod}', API='{sel_api}'")
                agent_config_to_modify['provider'] = sel_prov
                agent_config_to_modify['model'] = sel_mod
                agent_config_to_modify['api'] = sel_api

        # For agent mode, we use the context manager to handle connections.
        async with runner:
            # This is the main agent chat loop, running inside the context
            print(f"\n--- Starting chat with {sel_agent} ---")
            print("Type your message. Use '/send' on a new line to submit. Type '/exit' to quit.")
            
            while True:
                user_input_text, local_image_path, image_url, should_exit = multiline_input()
                if should_exit:
                    break
                if not user_input_text and not local_image_path and not image_url:
                    continue

                # Construct multimodal content
                user_message_content = []
                if user_input_text:
                    user_message_content.append({"type": "input_text", "text": user_input_text})
                
                if local_image_path:
                    image_data = get_image_base64(local_image_path)
                    if image_data:
                        user_message_content.append({"type": "image_base64", "image_base64": image_data})
                    else:
                        print(f"❌ Could not process local image from path: {local_image_path}. Skipping image.")
                        if not user_input_text and not image_url: continue

                if image_url:
                    user_message_content.append({"type": "image_url", "image_url": {"url": image_url}})

                if not user_message_content: continue

                try:
                    result = await runner.run(agent_name=sel_agent, user_message=user_message_content)
                    
                    print(f"Assistant > {result}\n")

                    text_for_token_estimation = user_input_text if user_input_text else ""
                    tok_count = estimate_tokens([
                        {"role": "user", "content": text_for_token_estimation},
                        {"role": "assistant", "content": result}
                    ])
                    print(f"\U0001F4CC Approx. token usage (last turn): {tok_count}\n")

                except Exception as e:
                    log_exception("An error occurred while running the agent", e)
                    print(f"❌ An error occurred: {e}")

            # After the loop, save the chat history.
            if sel_agent in runner.main_history and len(runner.main_history[sel_agent]) > 0:
                logging.info(f"Saving chat log for agent '{sel_agent}'.")
                history_to_save = runner.main_history[sel_agent]
                
                json_log_path = save_chat_log_json(history_to_save, f"agent_{sel_agent}", sel_prov, sel_mod, chat_log_dir)
                txt_log_path = save_chat_log_txt(history_to_save, f"agent_{sel_agent}", sel_prov, sel_mod, chat_log_dir)
                
                print(f"💾 Chat logs saved to {json_log_path} and {txt_log_path}")
            else:
                logging.info("No history to save for this agent session.")

    except Exception as e:
        log_exception("A critical error occurred in the main application", e)

    logging.info("Session ended.")


if __name__ == "__main__":
    asyncio.run(main())

