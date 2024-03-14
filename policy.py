from enum import Enum
import numpy as np

from system import states, to_idx, get_valid_actions


class PolicyInit(Enum):
    GIVEN = 1
    RANDOM =  2
    DETERMINISTIC = 3
    EQUAL = 4
    ZERO = 5


class Policy:
    def __init__(
        self,
        num_states: int,
        num_actions: int,
        init=PolicyInit.RANDOM,
        given_policy: np.ndarray | None = None,
    ):
        self.num_states = num_states
        self.num_actions = num_actions
        self.sa = np.zeros(num_states, dtype=int)
        if init is PolicyInit.RANDOM:
            self.policy = np.random.rand(self.num_states, self.num_actions)
            self.purge_invalid_actions()
            self.policy = (
                self.policy / np.sum(self.policy, axis=1)[:, None]
            )  # Normalize
        elif init is PolicyInit.EQUAL:
            self.policy = np.ones((self.num_states, self.num_actions))
            self.purge_invalid_actions()
            self.policy = (
                self.policy / np.sum(self.policy, axis=1)[:, None]
            )  # Normalize
        elif init is PolicyInit.DETERMINISTIC:
            self.policy = np.zeros((self.num_states, self.num_actions))
            for s in self.policy:
                s[np.random.choice(get_valid_actions(s))] = 1
        elif init is PolicyInit.GIVEN:
            assert given_policy is not None
            self.validate(given_policy)
            self.policy = given_policy
        elif init is PolicyInit.ZERO:
            self.policy = np.zeros((self.num_states, self.num_actions))
        else:
            raise ValueError("Invalid policy initialization")

    def prob(self, state: np.ndarray, action: int | None = None) -> np.ndarray | float:
        if action is None:
            return self.policy[to_idx(state)]
        return self.policy[to_idx(state), action]

    def update(self, state: np.ndarray, action: int, prob) -> None:
        self.policy[to_idx(state), action] = prob

    def gen_action_idx(self, state: np.ndarray) -> int:
        return np.random.choice(self.num_actions, p=self.prob(state))
    
    def get_action(self, state: np.ndarray) -> int:
        return self.sa[to_idx(state)]
    
    def set_action(self, state: np.ndarray, action: int) -> None:
        self.sa[to_idx(state)] = action

    def purge_invalid_actions(self, policy=None) -> None:
        for s in states:
            valid_actions_idx = set(get_valid_actions(s, idx=True))
            for a in range(self.num_actions):
                if a not in valid_actions_idx:
                    if(policy is not None):
                        policy[to_idx(s), a] = 0
                    else:
                        self.policy[to_idx(s), a] = 0

    def validate(self, policy = None) -> None:
        if(policy is None):
            policy = self.policy
        self.purge_invalid_actions(policy)
        if not policy.shape[0] == self.num_states:
            raise ValueError("Given policy does not match the number of states")
        if not policy.shape[1] == self.num_actions:
            raise ValueError("Given policy does not match the number of actions")
        if not np.all(np.sum(policy, axis=1) == 1):
            raise ValueError("Given policy is not normalized")
