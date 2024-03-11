from system import mdp, reward, num_states, num_actions
from model import Model
from policy import Policy, PolicyInit

model = Model(mdp, reward)

policy = Policy(num_states, num_actions, PolicyInit.DETERMINISTIC)
