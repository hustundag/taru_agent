import asyncio
import logging
import os
import shutil
import signal
import subprocess
from datetime import datetime
from TaruAgent import TaruRunner
import sys

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
    test_name = "T6_mcp"
    setup_logging()
    mcp_server_process = None
    try:
        print(f"--- Starting Test Case: {test_name} ---")
        # 1. Start the MCP server in the background
        logging.info("Starting MCP server...")
        mcp_server_process = subprocess.Popen(
            [sys.executable, "mcp_servers/taru_mcp_files.py"], # Use sys.executable here
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            preexec_fn=os.setsid
        )
        await asyncio.sleep(3) # Give the server a moment to start

        if mcp_server_process.poll() is not None:
            stderr_output = mcp_server_process.stderr.read().decode()
            logging.error(f"Failed to start MCP server. Error:\n{stderr_output}")
            print(f"--- {test_name} FAILED: MCP server failed to start. ---")
            return

        logging.info(f"MCP Server started with PID: {mcp_server_process.pid}")

        # 2. Run the agent that uses an MCP tool
        async with TaruRunner(config_path="config_T6.yaml") as runner:
            if not os.path.exists("data"): os.makedirs("data")
            with open("data/mcp_test_file.txt", "w") as f:
                f.write("hello from mcp test")
            
            logging.info("Running agent to list files via MCP...")
            response = await runner.run(
                "MCP_File_Agent", 
                user_message=[{"type": "input_text", "text": "List the files in the root directory."}]
            )
            print(f"\nAgent Response: {response}")

            # 3. Verification
            if "mcp_test_file.txt" in response:
                print(f"--- {test_name} PASSED ---")
            else:
                print(f"--- {test_name} FAILED: Did not find the test file via MCP. ---")

    except Exception as e:
        logging.error(f"Test {test_name} failed with an exception: {e}", exc_info=True)
        print(f"--- {test_name} FAILED with exception. ---")
    finally:
        if mcp_server_process:
            logging.info(f"Terminating MCP server (PID: {mcp_server_process.pid})...")
            os.killpg(os.getpgid(mcp_server_process.pid), signal.SIGTERM)
            mcp_server_process.wait()
            logging.info("MCP server terminated.")
        archive_log_file(test_name)

if __name__ == "__main__":
    asyncio.run(main())
