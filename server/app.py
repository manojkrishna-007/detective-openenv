from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from env import DetectiveEnv

app = FastAPI()

env_instance = None


class ActionInput(BaseModel):
    action: str


@app.get("/")
def root():
    return {
        "message": "Detective OpenEnv is running",
        "endpoints": {
            "reset": "/reset (POST)",
            "step": "/step (POST)",
            "state": "/state (GET)"
        }
    }


@app.post("/reset")
def reset():
    global env_instance

    env_instance = DetectiveEnv(difficulty="easy")

    obs = env_instance.reset()

    return {
        "observation": obs.dict(),
        "done": False
    }


@app.post("/step")
def step(action: ActionInput):
    global env_instance

    obs, reward, done, info = env_instance.step(action)

    return {
        "observation": obs.dict(),
        "reward": reward.value,
        "done": done,
        "info": info
    }


@app.get("/state")
def state():
    return env_instance.state()


# ======================
# REQUIRED FOR VALIDATOR
# ======================
def main():
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
