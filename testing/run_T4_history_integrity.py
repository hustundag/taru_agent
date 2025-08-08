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

async def main():
    test_name = "T4_history_integrity"
    setup_logging()
    try:
        print(f"--- Starting Test Case: {test_name} ---")
        agent_name = "Jump_Test_Agent"
        
        async with TaruRunner(config_path="config_T4.yaml") as runner:
            # --- Test 1: Initial message and echo --- #
            print("\nStep 1: Testing initial echo...")
            response1 = await runner.run(agent_name, user_message=[{"type": "input_text", "text": "How are you today"}])
            print(f"Agent Response 1: {response1}")
            if response1 != "How are you today is echoed":
                print("TEST FAILED: Initial echo did not match.")
                return
            print("Step 1 PASSED.")

            # --- Test 2: Intercept at_function_call --- #
            print("\nStep 2: Testing jump from at_function_call...")
            response2 = await runner.run(agent_name, user_message=[{"type": "input_text", "text": "What is 1 + 1?"}])
            print(f"Agent Response 2: {response2}")
            if response2 != "intercepted at function call":
                print("TEST FAILED: Did not intercept at function call.")
                return
            print("Step 2 PASSED.")

            # --- Test 3: Intercept at_function_call_return --- #
            print("\nStep 3: Testing jump from at_function_call_return...")
            response3 = await runner.run(agent_name, user_message=[{"type": "input_text", "text": "What are the gas prices?"}])
            print(f"Agent Response 3: {response3}")
            if response3 != "intercepted at function call return":
                print("TEST FAILED: Did not intercept at function call return.")
                return
            print("Step 3 PASSED.")

            # --- Test 4: Coherency Check --- #
            print("\nStep 4: Testing LLM coherency after jumps...")
            response4 = await runner.run(agent_name, user_message=[{"type": "input_text", "text": "How are you today"}])
            print(f"Agent Response 4: {response4}")
            if response1 != "How are you today is echoed":
                print("TEST FAILED: LLM did not recover and call the tool correctly.")
                return
            print("Step 4 PASSED.")

        print(f"\n--- {test_name} PASSED ---")
    except Exception as e:
        logging.error(f"Test {test_name} failed with an exception: {e}", exc_info=True)
        print(f"--- {test_name} FAILED with exception. ---")
    finally:
        archive_log_file(test_name)

if __name__ == "__main__":
    asyncio.run(main())