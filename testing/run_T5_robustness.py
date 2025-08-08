import asyncio
import logging
import os
import shutil
from datetime import datetime
from TaruAgent import TaruRunner

LOG_FILE = "taru_agent.log"

def setup_logging():
    if os.path.exists(LOG_FILE):
        os.remove(LOG_FILE)
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        handlers=[
            logging.FileHandler(LOG_FILE),
            logging.StreamHandler()
        ]
    )
    logging.info("Logging initialized.")

def archive_log_file(test_case_name):
    if not os.path.exists(LOG_FILE):
        return
    logs_dir = "test_logs"
    os.makedirs(logs_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    new_log_filename = f"{timestamp}_{test_case_name}.log"
    destination = os.path.join(logs_dir, new_log_filename)
    shutil.copy(LOG_FILE, destination)
    logging.info(f"Archived log file to {destination}")

async def run_test(test_name, agent_name, config_path="config_T5.yaml"):
    print(f"\n--- Running Robustness Test: {test_name} ---")
    test_case_name_for_log = f"T5_robustness_{test_name.replace(':', '-').replace(' ', '_')}"
    setup_logging()
    try:
        if test_name == "5.9: Syntax Error in Tool File":
            try:
                async with TaruRunner(config_path=config_path) as runner:
                    pass
                print(f"--- Test '{test_name}' FAILED: TaruRunner initialized without error. ---")
            except Exception as e:
                logging.info(f"Successfully caught expected error: {e}")
                print(f"--- Test '{test_name}' Completed Without Crashing ---")
            return

        async with TaruRunner(config_path=config_path) as runner:
            response = await runner.run(agent_name, user_message=[{"type": "input_text", "text": "Start"}])
            print(f"Agent Response: {response}")
            print(f"--- Test '{test_name}' Completed Without Crashing ---")
    except Exception as e:
        logging.error(f"Test {test_name} failed with an exception: {e}", exc_info=True)
        print(f"--- Test '{test_name}' FAILED with unexpected exception: {e} ---")
    finally:
        archive_log_file(test_case_name_for_log)

async def main():
    # 5.1: Max Turn Count Exceeded
    await run_test("5.1: Max Turn Count", "Max_Turn_Agent")

    # 5.2: Config file is missing the entire local_tools section
    broken_config_no_tools = """
providers:
  openai:
    api_key: "env:OPENAI_API_KEY"
    models:
      - name: "gpt-4.1-mini"
agents:
  - name: Dummy_Agent
    provider: openai
    model: gpt-4.1-mini
    instruction: "Hello"
"""
    with open("config_broken.yaml", "w") as f:
        f.write(broken_config_no_tools)
    await run_test("5.2: Missing local_tools section", "Dummy_Agent", config_path="config_broken.yaml")

    # 5.3: Tool in resources, but not in local_tools config
    await run_test("5.3: Missing Tool Definition", "Missing_Tool_Agent")

    # 5.4: Tool produces bad JSON
    await run_test("5.4: Tool Produces Bad JSON", "Bad_JSON_Tool_Agent")

    # 5.5: Action produces faulty JSON
    await run_test("5.5: Action Produces Bad JSON", "Bad_JSON_Action_Agent")

    # 5.6: Action produces non-existent field
    await run_test("5.6: Action Produces Bad Schema", "Bad_Field_Action_Agent")

    # 5.7: LLM hallucinates a tool call
    await run_test("5.7: LLM Hallucinates Tool", "Hallucination_Agent")

    # 5.9: Syntax Error in Tool File
    await run_test("5.9: Syntax Error in Tool File", "Syntax_Error_Agent")

    # 5.10: Invalid Custom Action Path
    await run_test("5.10: Invalid Custom Action Path", "Bad_Action_Path_Agent")

    # 5.11: Handoff to Non-Existent Agent
    await run_test("5.11: Handoff to Non-Existent Agent", "Bad_Handoff_Target_Agent")

    # 5.12: LLM provides incorrect tool arguments
    await run_test("5.12: LLM Provides Bad Tool Arguments", "Bad_Tool_Args_Agent")

if __name__ == "__main__":
    asyncio.run(main())
