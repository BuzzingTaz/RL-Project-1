import sys
import numpy as np

from system import (
    mdp,
    reward,
    states,
    num_states,
    num_actions,
    to_idx,
    get_valid_actions,
)
from model import Model
from policy import Policy, PolicyInit


def EvaluatePolicy(
    policy: Policy, valf: np.ndarray, model: Model, thresh=0.01, gamma=0.9
):
    i = 0
    delta = thresh + 1
    while delta > thresh and i < 1000:
        delta = 0
        for s in states:
            v = valf[to_idx(s)]
            a = policy.get_action(s)
            # policy.set_action(to_state(s), a)

            valf[to_idx(s)] = model.prob(s, a) @ (model.get_reward(s) + gamma * valf)
            delta = max(delta, abs(v - valf[to_idx(s)]))
        i += 1
    return valf


def UpdatePolicy(policy: Policy, valf: np.ndarray, model: Model, gamma=0.9):
    policy_stable = True
    k = 0
    for s in states:
        old_action = policy.get_action(s)
        amzt = np.argmax(
            [
                model.prob(s, a) @ (model.get_reward(s) + gamma * valf)
                for a in get_valid_actions(s, idx=True)
            ]
        )
        amzt = get_valid_actions(s, idx=True)[amzt]
        policy.set_action(s, amzt)
        if old_action != policy.get_action(s):
            k += 1
            policy_stable = False
    print(f"Policy changed for {k} states")
    return policy_stable


# Open a file for writing
with open("output.txt", "w") as file:
    # Redirect output to the file
    sys.stdout = file
    # Initialize model
    model = Model(mdp, reward)
    # Initialize policy
    policy = Policy(num_states, num_actions, PolicyInit.RANDOM)
    for s in states:  # Make sure action is valid
        policy.set_action(s, policy.gen_action_idx(s))
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
        for s in states:
            print(f"Value Function of State {s}: {valf[to_idx(s)]}")
        # Policy Improvement
        policy_stable = UpdatePolicy(policy, valf, model)
        # Print policy
        for s in states:
            print(f"Policy of State {s}: {policy.get_action(s)}")
        i += 1

    # Restore standard output
    sys.stdout = sys.__stdout__
