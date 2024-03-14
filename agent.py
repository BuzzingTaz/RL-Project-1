import numpy as np

from model import Model
from policy import Policy
from system import states, actions, wind_col, num_states, num_actions, init_mdp, init_reward, t_state


class Agent:
    def __init__(self, model: Model, policy: Policy):
        self.model = model
        self.policy = policy

    def change_policy(self, policy: Policy):
        self.policy = policy

    def gen_episode(self, start_state: np.ndarray, start_action: int, T: int):
        estate = [start_state]
        eaction = [start_action]
        ereward = [self.model.get_reward(start_state, self.model.gen_next(start_state, start_action))]
        for i in range(T):
           estate.append(self.model.gen_next(estate[-1], eaction[-1]))
           eaction.append(self.policy.get_action(estate[-1]))
           ereward.append(self.model.get_reward(estate[-2], estate[-1]))
           if(ereward==self.model.get_reward(estate[-2], t_state)):
               break

        return estate, eaction, ereward 

