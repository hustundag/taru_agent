# TaruAIcli

`TaruAIcli` is a command-line interface for interacting with the TaruAgent framework. It allows you to chat with different AI agents, using various providers and models, all from your terminal.

## Installation

1.  **Install Dependencies:**

    ```bash
    pip3 install -r requirements.txt
    ```

## Configuration

`TaruAIcli` uses a `config.yaml` file for configuration. Here's an example:

```yaml
# config.yaml

providers:
  openai:
    api_key: "env:OPENAI_API_KEY" # Use environment variable
    models:
      - "gpt-4.1-mini"
      - "o4-mini"
  google:
    api_key: "env:GOOGLE_API_KEY"
    models:
      - "gemini-2.5-flash"
  ollama:
    - name: "my_local_ollama_1"
      base_url: "http://10.21.2.101:11434"
      models:
        - "deepseek-r1:1.5b"
        - "llama3.2:1b"
    - name: "my_local_ollama_2"
      base_url: "http://10.21.2.102:11434"
      models:
        - "deepseek-r1:1.5b"
        - "llama3.2:1b"

# 2. Default Settings
defaults:
  agent: "TaruDefaultAgent"

runflow:
  use_default_agent: true


# 3. System Settings
system:
  logging:
    level: "DEBUG" # DEBUG, INFO, WARNING, ERROR
    chat_log_dir: "./chat_logs"
    log_format: ["text"] # Can be "text", "json", or a list: ["text", "json"]

4. Model Context Protocol (MCP) Server Definitions Examples 
mcp_servers:
  - name: "taru_sandbox_mcp"
    url: "http://10.21.2.142:8507/sse"
    description: "MCP server for interacting to list and read files."


# 5. Local Tool Discovery
local_tools:
  - name: "time_utils"
    path: "./custom_functions/time_utils/"
  - name: "file_system_utils"
    path: "./custom_functions/file_tools/"

    
# 6. Agent Definitions
agents:
  - name: "DefaultAgent"
    provider: "openai"
    model: "gpt-4.1-mini"
    #api: v1_chat_completions # Default is 'v1_chat_completions'. Options: 'v1_responses', 'gemini_v1beta', 'ollama_openai_compatible'
    # history_mode: full # Default is 'full'. Options: 'none', 'clean'. For 'v1_responses' API, you can also use 'response_id'.

  - name: "TaruDefaultAgent"
    provider: "openai"
    model: "gpt-4.1-mini"
    api: v1_responses
    history_mode: response_id
    model_settings: {}
    instruction: "You are a helpful general-purpose assistant. You are running in a test environment."
    max_turn_count: 100
    resources:
      local_tools:
        - group_name: "time_utils"
        - group_name: "file_system_utils"
```

*   **`providers`**: Define your LLM providers here.
*   **`agents`**: Define your agents here.
*   **`system`**: Configure logging and other system settings.
*   **`defaults`**: Set the default provider, model, and agent for the CLI.
*   **`runflow`**: Set to `true` to skip the interactive selection and use the default agent. 

## Usage

1.  **Run the CLI:**

    ```bash
    python3 TaruAIcli.py
    ```

2.  **Select Provider, Model, and Agent:**

    You will be prompted to select a provider, model, and agent from your `config.yaml` file.

3.  **Chat with the Agent:**

    *   Type your message and press `Enter`.
    *   To send a multiline message, end your input with `/send` on a new line.
    *   To send a local image, type `/image <path_to_image>`.
    *   To send an image from a URL, type `/image_url <url_of_image>`.
    *   To exit, type `/exit`.

