import json
from TaruAgent import tool

@tool("get_inventory_data", "Retrieves inventory data from multiple warehouses.")
def get_inventory_data() -> dict:
    """A local tool that reads chunked inventory data from a JSON file."""
    try:
        with open('./data/inventory_data.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {"error": "Inventory data file not found."}
    except json.JSONDecodeError:
        return {"error": "Invalid JSON in inventory data file."}
