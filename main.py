import sys
from system import mdp, reward, num_states, num_actions, to_idx, to_state
from model import Model
from policy import Policy, PolicyInit
import numpy as np

def EvaluatePolicy(policy, valf, model, thresh = 0.01):
    i = 0
    uvalf = valf.copy()
    delta = thresh + 1
    while delta > thresh and i < 1000:
        delta = 0
        for s in range(num_states):
            v = valf[s]
            uvalf[s] = 0
            a = policy.get_action(to_state(s))
            # policy.set_action(to_state(s), a)
            for s_ in range(num_states):
                uvalf[s] +=  model.prob(to_state(s), int(a), to_state(s_)) * (model.get_reward(to_state(s), to_state(s_)) + valf[s_])
            valf = uvalf
            delta = max(delta, abs(v - uvalf[s]))
        i += 1
    return uvalf
    
def UpdatePolicy(policy, valf, model):
    policy_stable = True
    for s in range(num_states):
        old_action = policy.get_action(to_state(s))
        amzt = np.argmax([sum(model.prob(to_state(s), int(a), to_state(s1)) * (model.get_reward(to_state(s), to_state(s1)) + valf[s1]) for s1 in range(num_states)) for a in range(num_actions)])
        policy.set_action(to_state(s), amzt)
        if old_action != policy.get_action(to_state(s)):
            policy_stable = False
    return policy_stable
# Open a file for writing
with open("output.txt", "w") as file:
    # Redirect output to the file
    sys.stdout = file
    # Initialize model
    model = Model(mdp, reward)
    # Initialize policy
    policy = Policy(num_states, num_actions, PolicyInit.RANDOM)
    for s in range(num_states):
        policy.set_action(to_state(s), policy.gen_action_idx(to_state(s)))
    # Initialize random state function
    valf = np.random.uniform(0, 2, size=num_states)
    thresh = 0.01
    policy_stable = False
    i = 0
    while (not policy_stable) and i < 1000:
        print(f"Policy Iteration {i + 1}")
        # Policy Evaluation
        valf = EvaluatePolicy(policy, valf, model, thresh)
        # Print valf
        for s in range(num_states):
            print(f"Value Function of State {s}: {valf[s]}")
        # Policy Improvement
        policy_stable = UpdatePolicy(policy, valf, model)
        # Print policy
        for s in range(num_states):
            print(f"Policy of State {s}: {policy.get_action(to_state(s))}")
        i += 1
    
    # Restore standard output
    sys.stdout = sys.__stdout__



