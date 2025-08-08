from TaruAgent import ActionOutput
import json

class BadActionOutput:
    """A class that looks like ActionOutput but isn't."""
    def __init__(self, next_action):
        self.next_action = next_action

def produce_faulty_json_action(state_data):
    """Returns an ActionOutput with a payload that is not valid JSON."""
    # This is a valid Python string literal, but its content is malformed JSON (missing closing brace)
    return ActionOutput(next_action="return_to_user", payload='{"key": "value", "missing_brace": "oops" ')

def produce_non_existent_field_action(state_data):
    """Returns an object that is not a valid ActionOutput schema."""
    # This will likely cause an AttributeError in the PolicyManager
    return BadActionOutput(next_action="return_to_user")