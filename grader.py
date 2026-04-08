def grade(info, total_reward, steps_used):

    score = 0.0

    if info.get("correct"):
        score += 0.6

    if info.get("evidence_strength", 0) >= 1.5:
        score += 0.2

    if steps_used <= 4:
        score += 0.2
    elif steps_used <= 6:
        score += 0.1

    return min(score, 1.0)