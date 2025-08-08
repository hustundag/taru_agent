import json
from TaruAgent import tool

@tool("get_log_data", "Retrieves system logs for analysis.")
def get_log_data() -> dict:
    """A local tool that reads chunked log data from a JSON file."""
    try:
        with open('./data/log_data.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {"error": "Log data file not found."}
    except json.JSONDecodeError:
        return {"error": "Invalid JSON in log data file."}
