import numpy as np
from system import to_idx, to_state


class Model:
    def __init__(self, model: np.ndarray, reward: np.ndarray):
        self.validate_model(model)
        self.mdp = model
        self.validate_reward(reward)
        self.reward = reward

    # Get probabilities of next state given a state and action or
    # Get probability of next state given a state, action and next state
    def prob(self, state: list, action_idx: int, next_state=None) -> np.ndarray | float:
        if next_state is None:
            return self.mdp[to_idx(state), action_idx]

        return self.mdp[to_idx(state), action_idx, to_idx(next_state)]

    # Get rewards to all next states or
    # Get reward to a specific next state
    def get_reward(self, state: list, next_state=None) -> np.ndarray | float:
        if next_state is None:
            return self.reward[to_idx(state)]

        return self.reward[to_idx(state), to_idx(next_state)]

    # Generate next state given a state and action
    def gen_next(self, state: list, action_idx: int) -> np.ndarray:
        return to_state(np.random.choice(self.mdp[to_idx(state), action_idx:int]))

    def validate_model(self, model: np.ndarray) -> None:
        if model.ndim != 3:
            raise ValueError("Model must be a 3D array")
        if model.sum(axis=2).all() != 1:
            raise ValueError("Model must be a probability matrix")

    def validate_reward(self, reward: np.ndarray) -> None:
        if reward.ndim != 2:
            raise ValueError("Reward must be a 2D array")
        if reward.shape[0] != reward.shape[1]:
            raise ValueError("Reward must be a square matrix")
        if reward.shape[0] != self.mdp.shape[0]:
            raise ValueError("Reward and model must have the same number of states")
