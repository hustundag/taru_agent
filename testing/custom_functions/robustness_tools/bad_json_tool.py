from TaruAgent import tool

@tool("produces_bad_json", "A tool that returns malformed JSON.")
def produces_bad_json() -> str:
    return '{"key": "value", "missing_quote: "oops"}'
