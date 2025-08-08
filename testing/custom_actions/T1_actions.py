from TaruAgent import ActionOutput

# 1) MATH GUARDRAIL
# Fails unless message contains "math"
def guardrail_math(data):
    text = str(data).lower()
    if "math" in text:
        return ActionOutput(is_guardrail=True, guardrail_result="passed")
    return ActionOutput(is_guardrail=True, guardrail_result="failed", payload="GUARDRAIL_MATH_FAILED")

# 2a) Inject date query after math answer
def check_initial_answer(data):
    # Tests jump from at_function_call to at_llm_final_response (func: get_date)
    if data == "The answer is 4.":
        return ActionOutput(payload="what is the date today", next_action="jump_to_return_to_llm", payload_replace=True)

    if data == "I don't know the date":
        return ActionOutput(payload="How is the weather today", next_action="jump_to_return_to_llm", payload_replace=True)

    # Tests jump from at_function_call to at_function_call_return func:suggest_dinner)
    if data == "Log data suggests that weather is sunny": 
        return ActionOutput(payload="what will we have for dinner", next_action="jump_to_return_to_llm", payload_replace=True)

    # Tests jump from at_function_call_return to at_llm_final_response  func: check_gas_prices
    if isinstance(data, str) and "we will have steak for dinner".lower() in data.lower():
        return ActionOutput(payload="what is gas price today", next_action="jump_to_return_to_llm", payload_replace=True)
    return ActionOutput()

# 2b) Illegal jump to FUNCTION_CALL
def illegal_jump_trigger(data):
    return ActionOutput(next_action="jump_to_function_call")

# 2c) No-ops action for testing next_action without any other options. 
def loop_breaker(data):
    if data == "Average gas price for the location is $5.28":
        return ActionOutput(next_action="continue")
    return ActionOutput()

# 3) Intercept date_time_finder calls
def intercept_date_call(data):
    return ActionOutput(payload="I don't know the date", next_action="jump_to_llm_final_response", payload_replace=True)

# 4) After weather returns, ask for log analysis
def check_weather_response(data):
    if data == "weather is sunny today":
        return ActionOutput(payload="Weather logs is now available in the logs. can you get final weather using tool handoff_to_LogAnalyzerAgent and let me know the outcome", payload_replace=True)
    return ActionOutput()

# 5) Intercept dinner calls
def intercept_dinner_call(data):
    return ActionOutput(payload="we will have steak for dinner", next_action="jump_to_function_call_return", payload_replace=True)

# 6: Intercept gas price return and chain to a jump
def intercept_gas_price_return(data):
    return ActionOutput(payload="Average gas price for the location is $5.28", next_action="execute_custom_function", target_path="custom_actions.T1_actions.perform_jump_to_final_response", payload_replace=True)

# 7: Performs the jump to AT_LLM_FINAL_RESPONSE
def perform_jump_to_final_response(data):
    return ActionOutput(payload=data, next_action="jump_to_llm_final_response", payload_replace=True)

# 8)if We hit here tests are mostly completed. just testing replace only action 
def end_tests(data):
    if data == "Average gas price for the location is $5.28":
        return ActionOutput(payload="This is the end of the tests", payload_replace=True)
    return ActionOutput()

# ) Final payload verify at user return
def modify_payload(data):
    alt_data= "Tests Completed Successfuly"
    if data == "This is the end of the tests":
        return ActionOutput(is_guardrail=True, guardrail_result="failed", payload=alt_data)
    return ActionOutput(is_guardrail=True, guardrail_result="passed", payload=data)

# Sub-agent no-op final
def noop_final(data):
    return ActionOutput()
