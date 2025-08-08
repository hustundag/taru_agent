from TaruAgent import tool

@tool("get_package_id", "Retrieves the package ID for a given order ID.")
def get_package_id(order_id: str) -> str:
    if order_id == "ORD-001":
        return "PKG-789"
    else:
        return "Package ID not found."
