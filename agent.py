import numpy as np

from model import Model
from policy import Policy
from system import (
    actions,
    t_state,
)


class Agent:
    def __init__(self, model: Model, policy: Policy):
        self.model = model
        self.policy = policy

    def change_policy(self, policy: Policy):
        self.policy = policy

    def gen_episode(self, start_state: np.ndarray, start_action: int, T: int):
        estate = [start_state]
        eaction = [start_action]
        ereward = [
            self.model.get_reward(
                start_state, self.model.gen_next(start_state, start_action)
            )
        ]
        for i in range(T):
            estate.append(self.model.gen_next(estate[-1], eaction[-1], astuple=True))
            eaction.append(self.policy.get_action(estate[-1]))
            ereward.append(self.model.get_reward(estate[-2], estate[-1]))
            if ereward[-1] == self.model.get_reward(estate[-2], t_state):
                break
        return estate, eaction, ereward

    def play(
        self, model: Model, policy: Policy, start_state: np.ndarray, max_steps: int
    ):
        # Run agent
        score = 0
        steps = 0
        s = start_state

        path = [s]

        while steps < max_steps:
            a = policy.get_action(s)

            s_ = model.gen_next(s, a)
            path.append(s_)

            r = model.get_reward(s, s_)
            score += r

            print(f"State: {s}, Action: {actions[a]}, Next State: {s_}, Reward: {r}")

            if r != -1:
                print(f"Game Over - Score: {score}")
                break

            s = s_
            steps += 1
        if(steps == max_steps):
            print(f"Game Over, Did not terminate - Score: {score}")
        return path, score
