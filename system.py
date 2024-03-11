import numpy as np

# System description

# States: grid coordinates
rows = 7
cols = 10
num_states = rows * cols

# State index conversion
def to_idx(state : list):
    return state[0] * cols + state[1]

def to_state(idx : int):
    return np.array([idx // cols, idx % cols])

# For episode generation 
def gen_random_state():
    return to_state(np.random.randint(0, num_states))

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
num_actions = actions.shape[0]

# Model: MDP
mdp = np.zeros((num_states, actions.shape[0], num_states))

def add_transition(mdp : np.ndarray, state_idx : int, action_idx : int, next_state : list, weight=1.0):
    next_state[0] = max(0, min(next_state[0], rows - 1))
    next_state[1] = max(0, min(next_state[1], cols - 1))
    next_idx = to_idx(next_state)
    mdp[state_idx, action_idx, next_idx] += weight

for idx in range(num_states):
    state = to_state(idx)
    for action_idx, action in enumerate(actions):
        if(wind_col[state[1]] != 0):
            wind = wind_col[state[1]]
            for noise in [-1, 0, 1]:
                next_state = state + action + (- np.array([noise + wind, 0])) # Negative because the wind goes up
                add_transition(mdp, idx, action_idx, next_state, weight=1/3)
        else:
            next_state = state + action
            add_transition(mdp, idx, action_idx, next_state)

 
# Reward: One dimensional
# -1 for each step, 0 for the goal 
reward = np.full((num_states, num_states), -1)
reward[:,to_idx([3, 7])] = 0