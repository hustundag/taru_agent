import json
from TaruAgent import tool

@tool("get_customer_id", "Retrieves the customer ID for a given customer name from a lookup file.")
def get_customer_id(customer_name: str) -> str:
    """A local tool that reads a customer ID from a JSON file based on customer name."""
    try:
        with open('./data/customer_lookup.json', 'r') as f:
            customer_data = json.load(f)
            return customer_data.get(customer_name, "Customer ID not found.")
    except FileNotFoundError:
        return "Error: Customer data file not found."
    except json.JSONDecodeError:
        return "Error: Invalid JSON in customer data file."
