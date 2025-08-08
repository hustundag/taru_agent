import json
from TaruAgent import tool

@tool(name="get_shipping_address", description="Retrieves the shipping address for a given package ID.")
def get_shipping_address(package_id: str) -> str:
    """
    Retrieves a mock shipping address based on the package ID.
    """
    # In a real scenario, this would query a database or external service.
    # For this test, we return a fixed mock address.
    return "123 Main St, Anytown, USA"