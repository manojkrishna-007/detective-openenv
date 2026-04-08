from pydantic import BaseModel
from typing import List, Dict


class Observation(BaseModel):
    clues: List[Dict]
    steps_left: int
    case_summary: str


class Action(BaseModel):
    action: str
    reasoning: str = ""


class Reward(BaseModel):
    value: float


class DetectiveEnv:
    def __init__(self, difficulty="easy"):
        self.difficulty = difficulty
        self.case_bank = {
            "easy": {
                "crime": "Financial fraud",
                "culprit": "bhavna",
                "clues": [
                    {"type": "fingerprint", "suspect": "bhavna", "reliability": 0.9},
                    {"type": "witness", "suspect": "bhavna", "reliability": 0.8},
                    {"type": "cctv", "suspect": "aarav", "reliability": 0.4},
                ],
                "max_steps": 6
            },
            "medium": {
                "crime": "Data breach",
                "culprit": "aarav",
                "clues": [
                    {"type": "server_log", "suspect": "aarav", "reliability": 0.7},
                    {"type": "badge_access", "suspect": "bhavna", "reliability": 0.6},
                    {"type": "usb_trace", "suspect": "aarav", "reliability": 0.8},
                ],
                "max_steps": 7
            },
            "hard": {
                "crime": "Insider leak",
                "culprit": "karan",
                "clues": [
                    {"type": "email", "suspect": "karan", "reliability": 0.6},
                    {"type": "fake_witness", "suspect": "aarav", "reliability": 0.85},
                    {"type": "access_log", "suspect": "karan", "reliability": 0.7},
                ],
                "max_steps": 8
            }
        }

        self.reset()

    def reset(self):
        case = self.case_bank[self.difficulty]

        self._state = {
            "clues_found": [],
            "index": 0,
            "steps_left": case["max_steps"],
            "done": False,
            "history": [],
            "case": case
        }

        return self._get_obs()

    def step(self, action: Action):

        if self._state["done"]:
            return self._get_obs(), Reward(value=0.0), True, {}

        act = action.action.lower()
        reward = 0
        info = {}

        case = self._state["case"]

        if act == "search_location":
            if self._state["index"] < len(case["clues"]):
                clue = case["clues"][self._state["index"]]
                self._state["clues_found"].append(clue)
                self._state["index"] += 1
                reward = 9
            else:
                reward = -2

        elif act == "analyze_evidence":
            if len(self._state["clues_found"]) < 2:
                reward = -2
            else:
                reward = 4

        elif act.startswith("accuse_"):
            suspect = act.split("_")[1]
            self._state["done"] = True

            evidence_strength = sum(
                c["reliability"]
                for c in self._state["clues_found"]
                if c["suspect"] == suspect
            )

            info["evidence_strength"] = evidence_strength

            if suspect == case["culprit"]:
                reward = 100
                info["correct"] = True
            else:
                reward = -100
                info["correct"] = False

        else:
            reward = -5

        self._state["history"].append(act)
        self._state["steps_left"] -= 1

        if self._state["steps_left"] <= 0:
            self._state["done"] = True

        return self._get_obs(), Reward(value=reward), self._state["done"], info

    def _get_obs(self):
        return Observation(
            clues=self._state["clues_found"],
            steps_left=self._state["steps_left"],
            case_summary=self._state["case"]["crime"]
        )

    def state(self):
        return {
            "clues_found": self._state["clues_found"],
            "steps_left": self._state["steps_left"],
            "done": self._state["done"]
        }