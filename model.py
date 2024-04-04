import numpy as np
from system import to_idx, to_state, num_states, cols


class Model:
    def __init__(self, model: np.ndarray, reward: np.ndarray):
        self.validate_model(model)
        self.mdp = model
        self.validate_reward(reward)
        self.reward = reward

    # Get probabilities of next state given a state and action or
    # Get probability of next state given a state, action and next state
    def prob(
        self, state: np.ndarray, action_idx: int, next_state: np.ndarray | None = None
    ) -> np.ndarray | float:
        if next_state is None:
            return self.mdp[to_idx(state), action_idx]
        return self.mdp[to_idx(state), action_idx, to_idx(next_state)]

    # Get rewards to all next states or
    # Get reward to a specific next state
    def get_reward(
        self, state: np.ndarray, next_state: np.ndarray | None = None
    ) -> np.ndarray | float:
        if next_state is None:
            return self.reward[to_idx(state)]

        return self.reward[to_idx(state), to_idx(next_state)]

    # Generate next state given a state and action
    def gen_next(self, state: np.ndarray, action_idx: int, astuple=False) -> np.ndarray:
        s_idx = np.random.choice(num_states, p=self.prob(state, action_idx))
        if(astuple):
           return (s_idx//cols, s_idx%cols) 
        return to_state(s_idx)

    def validate_model(self, model: np.ndarray) -> None:
        if model.ndim != 3:
            raise ValueError("Model must be a 3D array")
        # if model.sum(axis=2).all() != 1:
        #     raise ValueError("Model must be a probability matrix")

    def validate_reward(self, reward: np.ndarray) -> None:
        if reward.ndim != 2:
            raise ValueError("Reward must be a 2D array")
        if reward.shape[0] != reward.shape[1]:
            raise ValueError("Reward must be a square matrix")
        if reward.shape[0] != self.mdp.shape[0]:
            raise ValueError("Reward and model must have the same number of states")
