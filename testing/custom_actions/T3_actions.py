from TaruAgent import ActionOutput


def verify_init_handoff(state_data):
    """Verifies the different stages of the handoff test flow."""
    if isinstance(state_data, str) and "Alice Corp" in state_data and "CUST-42" in state_data :
        print("Policy-driven handoff PASSED!")
        return ActionOutput(
            next_action="continue",
            payload="Find the shipping status for order ORD-001.",
            payload_replace=True
        )
    else:
        print(f"Handoff verification FAILED. Unexpected result: {state_data}")
        return ActionOutput(next_action="jump_to_user_return",
        payload= "Handoff at_user_message FAILED",
        payload_replace=True)

def verify_handoff_flow(state_data):
    """Verifies the different stages of the handoff test flow."""
    if isinstance(state_data, str) and "ORD-001" in state_data and "SHIPPED" in state_data and "123 Main St, Anytown, USA" in state_data:
        print("Sequential handoff tool PASSED!")
        return ActionOutput(
            next_action="jump_to_return_to_llm",
            payload="Get the ID for 'Bob LLC' ",
            payload_replace=True
        )
    elif isinstance(state_data, str) and "Bob LLC" in state_data and "CUST-99" in state_data:
        print("Simple handoff tool PASSED!")
        return ActionOutput(next_action="jump_to_user_return", payload="All jump and tool handoff tests passed!", payload_replace=True)
    else:
        print(f"Handoff verification FAILED. Unexpected result: {state_data}")
        return ActionOutput(next_action="jump_to_user_return")

def nohit(state_data):
    return ActionOutput(next_action="jump_to_user_return", payload="if you hit here, test is failed !", payload_replace=True)