from TaruAgent import tool

@tool(name="get_weather", description="Get current weather for a location")
def get_weather(location: str) -> str:
    """Returns a dummy weather report for the given location."""
    return "weather is sunny today"
