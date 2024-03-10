import numpy as np
from system import to_idx, to_state

class Model:
    def __init__(self, model : np.ndarray, reward: np.ndarray):
        self.validate_model(model)
        self.model = model
        self.validate_reward(reward)
        self.reward = reward

    def validate_model(self, model : np.ndarray):
        if model.ndim != 2:
            raise ValueError("Model must be a 2D array")
    
    def validate_reward(self, reward : np.ndarray):
        if reward.ndim != 2:
            raise ValueError("Reward must be a 2D array")
        if reward.shape[0] != reward.shape[1]:
            raise ValueError("Reward must be a square matrix")
        if reward.shape[0] != self.model.shape[0]:
            raise ValueError("Reward and model must have the same number of states")
    
    def get_next_state(self, state, action):
        return to_state(np.random.choice(self.model[to_idx(state), action]))

    def get_reward(self, state, next_state):
        return self.reward[to_idx(state), to_idx(next_state)]
