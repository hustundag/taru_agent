from TaruAgent import tool

@tool(name="add_numbers", description="Add two numbers")
def add_numbers(a: int, b: int) -> int:
    """Returns the sum of a and b."""
    return int(a) + int(b)
