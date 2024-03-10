import numpy as np

# System description

# States: grid coordinates
rows = 7
cols = 10
total_states = rows * cols

# State index conversion
def to_idx(state : list):
    return state[0] * cols + state[1]

def to_state(idx : int):
    return np.array([idx // cols, idx % cols])


# Wind speeds
wind_col = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]

# For episode generation
def wind_effect(col : int):
    if(wind_col[col] == 0):
        return np.array([0, 0])
    return - np.array([wind_col + np.random.choice([-1, 0, 1]), 0])


# Actions: Kings moves
actions = np.array([
    [-1, -1],
    [-1, 0],
    [-1, 1],
    [0, -1],
    [0, 1],
    [1, -1],
    [1, 0],
    [1, 1]
])


# Model: MDP
mdp = np.empty((total_states, actions.shape[0]), dtype=object)
for idx in range(total_states):
    for action_idx in range(actions.shape[0]):
        mdp[idx, action_idx] = np.array([])

def add_transition(mdp : np.ndarray, state_idx, action_idx, next_state, weight=1):
    next_state[0] = max(0, min(next_state[0], rows - 1))
    next_state[1] = max(0, min(next_state[1], cols - 1))
    next_idx = to_idx(next_state)
    if(next_idx == state_idx):
        return mdp
    mdp[idx, action_idx] = np.append(mdp[idx, action_idx], next_idx)
    return mdp

for idx in range(total_states):
    state = to_state(idx)
    for action_idx, action in enumerate(actions):
        if(wind_col[state[1]] != 0):
            wind = wind_col[state[1]]
            for noise in [-1, 0, 1]:
                next_state = state + action + (- np.array([noise + wind, 0])) # Negative because the wind goes up
                mdp = add_transition(mdp, idx, action_idx, next_state)
        else:
            next_state = state + action
            mdp = add_transition(mdp, idx, action_idx, next_state)


# Reward: -1 for each step
reward = np.full((total_states, total_states), -1)