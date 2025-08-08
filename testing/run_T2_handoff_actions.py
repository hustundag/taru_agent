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
    test_name = "T2_handoff_actions"
    setup_logging()
    try:
        print(f"--- Starting Test Case: {test_name} ---")
        async with TaruRunner(config_path="config_T2.yaml") as runner:
            response = await runner.run(
                "Handoff_Initiator_Agent", 
                user_message=[{"type": "input_text", "text": "Please provide sales data for SKU-123."}]
            )
            print(f"\nFinal Agent Response: {response}")
            if response == "All policy-driven handoff tests passed successfully.":
                print(f"--- {test_name} PASSED ---")
            else:
                print(f"--- {test_name} FAILED: Unexpected final response. ---")
    except Exception as e:
        logging.error(f"Test {test_name} failed with an exception: {e}", exc_info=True)
        print(f"--- {test_name} FAILED with exception. ---")
    finally:
        archive_log_file(test_name)

if __name__ == "__main__":
    asyncio.run(main())