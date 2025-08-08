from TaruAgent import ActionOutput

def verify_data_analysis(data):
    """Receives the raw analysis data from the sub-agent and verifies it."""
    data_str = str(data).lower()
    # Check for sales data: sku-123 and 80
    if "sku-123" in data_str and "80" in data_str and "sales" in data_str:
        print("Handoff feed chunks test PASSED!")
        return ActionOutput(
            next_action="jump_to_return_to_llm",
            payload="Now, please get the inventory data for SKU-123.",
            payload_replace=True
        )
    # Check for inventory data: sku-123 and 350
    if "inventory" in data_str and "sku-123" in data_str and "350" in data_str:
         print("Handoff serial test PASSED!")
         return ActionOutput(
             next_action="jump_to_return_to_llm",
             payload="Great. Now, please analyze system logs and get me total number of errors.",
             payload_replace=True
         )
    # Check for error logs: error and 2
    if "error" in data_str and "2" in data_str and "the total number" in data_str:
         print("Handoff parallel test PASSED!")
         return ActionOutput(
             next_action="jump_to_user_return",
             payload="All policy-driven handoff tests passed successfully.",
             payload_replace=True
         )
    return ActionOutput(next_action="continue", payload=data)
