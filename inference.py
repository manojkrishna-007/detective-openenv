import os
from openai import OpenAI
from env import DetectiveEnv, Action
from tasks import TASKS
from grader import grade

API_BASE_URL = os.getenv("API_BASE_URL")
MODEL_NAME = os.getenv("MODEL_NAME")
API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("HF_TOKEN")

client = OpenAI(
    api_key=API_KEY,
    base_url=API_BASE_URL
)

BENCHMARK = "detective-openenv"
MAX_STEPS = 10

def log_start(task):
    print(f"[START] task={task} env={BENCHMARK} model={MODEL_NAME}", flush=True)


def log_step(step, action, reward, done, error):
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error}",
        flush=True
    )


def log_end(success, steps, score, rewards):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True
    )


def get_action(obs):
    """
    Uses OpenAI client for ALL decisions (mandatory compliance)
    Deterministic (temperature=0)
    """

    prompt = f"""
You are a professional detective AI.

Case:
{obs.case_summary}

Clues:
{obs.clues}

Steps remaining:
{obs.steps_left}

Rules:
- If fewer than 2 clues → search_location
- If clues strongly support one suspect → accuse
- If unsure → search_location
- Avoid unnecessary analyze_evidence unless helpful

Available actions:
search_location
analyze_evidence
accuse_aarav
accuse_bhavna
accuse_karan

Return ONLY the action name. No explanation.
"""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=20
        )

        action = (response.choices[0].message.content or "").strip().lower()

        valid_actions = {
            "search_location",
            "analyze_evidence",
            "accuse_aarav",
            "accuse_bhavna",
            "accuse_karan"
        }

        if action not in valid_actions:
            return "search_location"

        return action

    except Exception as e:
        print(f"[DEBUG] LLM error: {e}", flush=True)
        return "search_location"


def run_task(task_name, difficulty):

    env = DetectiveEnv(difficulty)
    obs = env.reset()

    rewards = []
    steps = 0
    success = False
    score = 0.0

    log_start(task_name)

    try:
        done = False
        info = {}

        while not done and steps < MAX_STEPS:

            action_text = get_action(obs)
            action = Action(action=action_text)

            obs, reward_obj, done, info = env.step(action)

            reward = reward_obj.value
            rewards.append(reward)
            steps += 1

            log_step(
                step=steps,
                action=action_text,
                reward=reward,
                done=done,
                error="null"
            )

        total_reward = sum(rewards)
        score = grade(info, total_reward, steps)
        success = score >= 0.5

    except Exception as e:
        log_step(step=steps, action="error", reward=0.0, done=True, error=str(e))

    finally:
        log_end(success, steps, score, rewards)
        print()

    return score



if __name__ == "__main__":

    scores = []

    for name, diff in TASKS.items():
        s = run_task(name, diff)
        scores.append(s)

    print("FINAL SCORE:", sum(scores) / len(scores))