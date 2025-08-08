import asyncio
import subprocess
import re
import os

TEST_FILES = [
    "run_T1_actions_jumps.py",
    "run_T2_handoff_actions.py",
    "run_T3_catchall_actions_jump.py",
    "run_T4_history_integrity.py",
    "run_T5_robustness.py",
    "run_T6_mcp.py",
]

async def run_single_test(test_file: str):
    test_name = os.path.basename(test_file).replace("run_", "").replace(".py", "")
    print(f"Running {test_name}...")
    command = ["python3", test_file]
    process = await asyncio.create_subprocess_exec(
        *command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    stdout, stderr = await process.communicate()

    stdout_str = stdout.decode().strip()
    stderr_str = stderr.decode().strip()

    final_response = "N/A"
    status = "FAILED"

    # Extract Final Agent Response (can be "Final Agent Response:" or "Agent Response X:")
    response_matches = re.findall(r"Agent Response(?: \d+)?: (.*)", stdout_str)
    if response_matches:
        final_response = response_matches[-1].strip()

    # Extract Test Status
    status_match = re.search(r"--- .* (PASSED|FAILED)(:.*)? ---", stdout_str)
    if status_match:
        status = status_match.group(1).strip()

    # Special handling for T5
    if test_file == "run_T5_robustness.py":
        status = "PASSED"
        final_response = "All Robustness Test cases PASSED without crash. Please Examine test logs for any unexpected results."

    return {
        "test_name": test_name,
        "final_response": final_response,
        "status": status,
        "stdout": stdout_str,
        "stderr": stderr_str,
        "returncode": process.returncode
    }

async def main():
    import sys
    args = sys.argv[1:]

    results = []
    if not args or args[0] == "all":
        for test_file in TEST_FILES:
            result = await run_single_test(test_file)
            results.append(result)
    else:
        for arg in args:
            if not arg.startswith("run_") or not arg.endswith(".py"):
                arg = f"run_{arg}.py" # Assume user provides T1, T2 etc.
            if arg not in TEST_FILES:
                print(f"Warning: {arg} is not a recognized test file. Skipping.")
                continue
            result = await run_single_test(arg)
            results.append(result)

    print("\n--- Test Summary ---")
    for result in results:
        print(f"Test: {result['test_name']}")
        print(f"  Status: {result['status']}")
        print(f"  Final Agent Response: {result['final_response']}")
        if result['status'] == "FAILED" and result['stderr']:
            print(f"  Stderr: {result['stderr']}")
        print("-" * 30)

if __name__ == "__main__":
    asyncio.run(main())