from enum import Enum
import numpy as np

from system import to_idx

class PolicyInit(Enum):
    GIVEN = 1
    RANDOM = 2
    DETERMINISTIC = 3
    EQUAL = 4

class Policy:
    def __init__(self, num_states : int, num_actions : int, init = PolicyInit.RANDOM, given_policy : np.ndarray | None = None):
        self.num_states = num_states
        self.num_actions = num_actions
        if init is PolicyInit.RANDOM:
            self.policy = np.random.rand(self.num_states, self.num_actions)
            self.policy = self.policy / np.sum(self.policy, axis=1)[:, None] # Normalize
        elif init is PolicyInit.EQUAL:
            self.policy = np.ones((self.num_states, self.num_actions))
            self.policy = self.policy / np.sum(self.policy, axis=1)[:, None] # Normalize 
        elif init is PolicyInit.DETERMINISTIC:
            self.policy = np.zeros((self.num_states, self.num_actions))
            for s in self.policy:
                s[np.random.randint(self.num_actions)] = 1
        elif init is PolicyInit.GIVEN:
            assert given_policy is not None
            self.validate(given_policy)
            self.policy = given_policy
        else:
            raise ValueError("Invalid policy initialization")
        
    
    def prob(self, state : list, action : int | None = None):        
        if action is None:
            return self.policy[to_idx(state)]
        
        return self.policy[to_idx(state), action]
    
    def update(self, state : list, action : int, prob):
        self.policy[to_idx(state), action] = prob

    def gen_action(self, state):
        return np.random.choice(self.num_actions, p=self.prob(state))

    def validate(self, policy : np.ndarray):
        if not policy.shape[0] == self.num_states:
            raise ValueError("Given policy does not match the number of states")
        if not policy.shape[1] == self.num_actions:
            raise ValueError("Given policy does not match the number of actions")
        if not np.all(np.sum(policy, axis=1) == 1):
            raise ValueError("Given policy is not normalized")