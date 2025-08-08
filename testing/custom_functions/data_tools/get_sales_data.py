import json
from TaruAgent import tool

@tool("get_sales_data", "Retrieves sales data from a file.")
def get_sales_data() -> dict:
    """A local tool that reads chunked sales data from a JSON file."""
    try:
        # Using a relative path for portability
        with open('./data/handoff_data.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {"error": "Data file not found."}
    except json.JSONDecodeError:
        return {"error": "Invalid JSON in data file."}
