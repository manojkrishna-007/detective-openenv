def grade(info, total_reward, steps_used):

    score = 0.0

    # correctness
    if info.get("correct"):
        score += 0.6

    # logical reasoning (from env)
    if info.get("logical"):
        score += 0.2

    # efficiency
    if steps_used <= 4:
        score += 0.2
    elif steps_used <= 6:
        score += 0.1

    # ======================
    # 🔒 STRICT RANGE FIX
    # ======================

    # lower bound (avoid 0.0)
    if score <= 0.0:
        score = 0.01

    # upper bound (avoid 1.0)
    if score >= 1.0:
        score = 0.99

    return score
