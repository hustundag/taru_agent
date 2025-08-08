from TaruAgent import tool

@tool("get_shipping_status", "Retrieves the shipping status for a given package ID.")
def get_shipping_status(package_id: str) -> str:
    if package_id == "PKG-789":
        return "SHIPPED"
    else:
        return "Status not found."
