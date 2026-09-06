def route_after_evaluate(state):
    MAX_ATTEMPTS = 3
    THRESHOLD = 0.8

    if state["faithfulness_score"] >= THRESHOLD:
        return "accept"
    elif state["attempt_count"] >= MAX_ATTEMPTS:
        return "give_up"
    else:
        return "retry"