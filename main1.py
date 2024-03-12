import sys
from system import mdp, reward, num_states, num_actions, to_idx, to_state
from model import Model
from policy import Policy, PolicyInit
import numpy as np

def Contraction(valf, model):
    delta = 0
    uvalf = valf.copy()
    for s in range(num_states):
        v = valf[s]
        uvalf[s] = np.max([sum(model.prob(to_state(s), int(a), to_state(s1)) * (model.get_reward(to_state(s), to_state(s1)) + valf[s1]) for s1 in range(num_states)) for a in range(num_actions)])        
        delta = max(delta, abs(v - uvalf[s]))
    return delta, uvalf
    
def GetPolicy(valf, model):
    policy = Policy(num_states, num_actions, PolicyInit.RANDOM)
    for s in range(num_states):
        for a in range(num_actions):
            policy.update(to_state(s), int(a), 0)
    for s in range(num_states):
        amzt = np.argmax([sum(model.prob(to_state(s), int(a), to_state(s1)) * (model.get_reward(to_state(s), to_state(s1)) + valf[s1]) for s1 in range(num_states)) for a in range(num_actions)])
        policy.update(to_state(s), int(amzt), 1)
        policy.set_action(to_state(s), amzt)
    return policy

# Open a file for writing
with open("output1.txt", "w") as file:
    # Redirect output to the file
    sys.stdout = file
    # Initialize model
    model = Model(mdp, reward)
    
    valf = np.zeros(num_states)
    thresh = 0.01
    delta = thresh + 1
    i = 0
    while delta > thresh and i < 1000:
        print(f"Value Iteration {i + 1}")
        # Apply contraction mapping
        delta, valf = Contraction(valf, model)
        # Print delta
        print(f"Delta: {delta}")
        # Print valf
        for s in range(num_states):
            print(f"Value Function of State {s}: {valf[s]}")
        i += 1
    policy = GetPolicy(valf, model)
    # Print policy
    for s in range(num_states):
        print(f"Policy of State {s}: {policy.get_action(to_state(s))}")
    
    # Restore standard output
    sys.stdout = sys.__stdout__



