import sys
import numpy as np

from system import (
    mdp,
    reward,
    states,
    num_states,
    num_actions,
    to_idx,
    to_state,
    get_valid_actions,
    init_mdp,
    init_reward,
    rows,
    cols,
)
from model import Model
from policy import Policy, PolicyInit


def EvaluateModel(
    model: Model, time_steps: int
):
    valf = np.zeros((time_steps + 1, num_states))
    for i in range(1, time_steps + 1):
        for s in range(num_states):
            valid_actions_idx = get_valid_actions(to_state(s), idx=True)
            valf[i][s] = np.max(
            [
                model.prob(to_state(s), a) @ (model.get_reward(to_state(s)) + valf[i - 1])
                for a in valid_actions_idx
            ]
        )
    return valf


def UpdatePolicy(policy: Policy, valf: np.ndarray, model: Model, time_step: int):
    policyt = Policy(num_states, num_actions, PolicyInit.RANDOM)
    k = 0
    for s in states:
        old_action = policy.get_action(s)
        valid_actions_idx = get_valid_actions(s, idx=True)
        amzt = np.argmax(
            [
                model.prob(s, a) @ (model.get_reward(s) + valf[time_step - 1])
                for a in valid_actions_idx
            ]
        )
        amzt = valid_actions_idx[amzt]
        policyt.set_action(s, amzt)
        if old_action != policyt.get_action(s):
            k += 1
    print(f"Policy changed for {k} states")
    return policyt


# Open a file for writing
with open("output.txt", "w") as file:
    # Redirect output to the file
    sys.stdout = file
    # Initialize model
    model = Model(mdp, reward)
    policies = []
    # Initialize policy
    policy = Policy(num_states, num_actions, PolicyInit.RANDOM)
    time_steps = 20
    for s in states:  # Make sure action is valid
        policy.set_action(s, policy.gen_action_idx(s))
    # Find Value function
    valf = EvaluateModel(model, time_steps)
    policy_stable = False
    i = 1
    while i < time_steps + 1:
        policies.append(policy)
        print(f"Value Iteration {i}")
        # Policy Evaluation
        valft = valf[i]
        # Print valf
        for s in states:
            print(f"Value Function of State {s}: {valft[to_idx(s)]}")
        # Policy Improvement
        policy = UpdatePolicy(policy, valf, model, i)
        # Print policy
        for s in states:
            print(f"Policy of State {s}: {policy.get_action(s)}")
        i += 1

    # Restore standard output
    sys.stdout = sys.__stdout__

import matplotlib.pyplot as plt

valf_toplot = valf[time_steps].reshape(rows, cols)
plt.gca().invert_yaxis()
heatmap = plt.imshow(valf_toplot)
plt.colorbar(heatmap)
plt.show()