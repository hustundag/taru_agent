from TaruAgent import ActionOutput

def intercept_at_function_call(state_data):
    """Intercepts the workflow at the function call stage and jumps to the user."""
    print("ACTION: Intercepting at function call.")
    return ActionOutput(
        next_action="jump_to_user_return",
        payload="intercepted at function call",
        payload_replace=True
    )

def intercept_at_function_call_return(state_data):
    """Intercepts the workflow at the function call return stage and jumps to the user."""
    print("ACTION: Intercepting at function call return.")
    return ActionOutput(
        next_action="jump_to_user_return",
        payload="intercepted at function call return",
        payload_replace=True
    )
