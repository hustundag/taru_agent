# TaruAgent Framework

**TaruAgent is a flexible, state-driven framework for building sophisticated, multi-agent enterprise applications. It is designed to address privacy and security concerns in AI adoption while managing the limited context size of Large Language Models (LLMs). You can use TaruAgent with leading providers like OpenAI and Google Gemini, as well as with your own local deployments via Ollama.**

For detailed documentation on architecture, advanced configuration, and features, please see the [**USER_GUIDE.md**](USER_GUIDE.md).

## ✨ Capabilities

The framework has the following set of features:

*   **🤖 Multi-Agent Orchestration:** Go beyond single-agent systems to organize and orchestrate model communication with your existing code.
*   **🤝 Advanced Handoff Strategies:** Natively supports multiple handoff flavors, including `serial`, `parallel`, and `sequential` pipelines, allowing agents to delegate tasks and collaborate effectively.
*   **🔌 Multi-Provider & Multi-Model Support:** Seamlessly switch between providers like **OpenAI**, **Google** APIs, or local deployments with **Ollama**.
*   **🔧 Local and MCP Tools:** Integrate any of your Python code as a local tool using a simple decorator. Supports remote tools utilizing the Model Context Protocol (MCP).
*   **📜 Policy-Driven State Machine:** A transparent state machine architecture allows you to implement complex logic, loops, and guardrails with a simple YAML configuration file.
*   **🚀 Simplified Setup with `TaruRunner`:** A high-level facade that simplifies initialization and execution.
*   **📝 Rich History Management:** Provides multiple history modes (`full`, `clean`, `none`, `response_id`) to control context and optimize for different use cases.

## 🚀 Quick Start

Get your first TaruAgent running in just a few minutes.

### 1. Prerequisites

*   Python 3.12+
*   `pip3` for package installation
*   An environment variable for your chosen LLM provider's API key (e.g., `OPENAI_API_KEY`).

### 2. Installation

This method installs the framework in editable mode, so any changes you make to the source are immediately available.

```bash
# Install the TaruAgent framework from the src directory
pip3 install -e src/

# Install other dependencies from the app's requirements file
pip3 install -r requirements.txt
```

### 3. Configuration (`config.yaml`)

Create a `config.yaml` file:

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

### 4. Create your Agent Runner (`run_agent.py`)

```python
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

```bash
# First, set your API key
export OPENAI_API_KEY="your_openai_api_key"

# Now, run your agent
python3 run_agent.py
```
