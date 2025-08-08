from TaruAgent import tool

@tool(name="get_date", description="Get current date and time")
def get_date(location: str) -> str:
    """Returns a dummy weather report for the given location."""
    return "march 15 2027"
