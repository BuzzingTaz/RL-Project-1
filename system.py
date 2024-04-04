import numpy as np

# System description

# States: grid coordinates
rows = 7
cols = 10
num_states = rows * cols
states = np.array([[i, j] for i in range(rows) for j in range(cols)])
t_state = np.array([3, 7])  # Terminal state
s_state = np.array([3, 0])  # Start state 


# State index conversion
def to_idx(state: np.ndarray | list | tuple) -> int:
    return state[0] * cols + state[1]


def to_state(idx: int) -> np.ndarray:
    return np.array([idx // cols, idx % cols])


# Wind speeds
wind_col = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]
# wind_col = [0] * cols


# For episode generation
def wind_effect(col: int) -> np.ndarray:
    if wind_col[col] == 0:
        return np.array([0, 0])
    return -np.array([wind_col[col] + np.random.choice([-1, 0, 1]), 0])


# Actions: Kings moves + Stay in place
actions = np.array(
    [
        [-1, -1],
        [-1, 0],
        [-1, 1],
        [0, -1],
        [0, 1],
        [1, -1],
        [1, 0],
        [1, 1],
        [0, 0],
    ]
)
num_actions = actions.shape[0]


# For episode generation
def gen_random_state() -> np.ndarray:
    return to_state(np.random.randint(0, num_states))


def gen_random_action(state: np.ndarray) -> int:
    return np.random.choice(get_valid_actions(state, idx=True))


def gen_random_sa() -> tuple[np.ndarray, int]:
    state = gen_random_state()
    action = gen_random_action(state)
    return state, action


def get_valid_actions(state: np.ndarray, idx: bool = False) -> np.ndarray:
    """Returns a list (or indices if idx=True) of valid actions for a given state.

    Args:
        state (np.ndarray): state
        idx (bool, optional): Whether to return indices. Defaults to False.

    Returns:
        np.ndarray: List of valid actions or indices
    """
    if (state == t_state).all():
        if idx:
            return np.array([num_actions - 1])
        return np.array([0, 0])

    valid_actions_idx = []
    for i in range(num_actions - 1):
        next_state = state + actions[i]
        if (
            (next_state[0] < 0)
            or (next_state[0] >= rows)
            or (next_state[1] < 0)
            or (next_state[1] >= cols)
        ):
            continue
        valid_actions_idx.append(i)
    if idx:
        return np.array(valid_actions_idx)
    return actions[valid_actions_idx]


# Model: MDP
def add_transition(
    mdp: np.ndarray, state_idx: int, action_idx: int, next_state: np.ndarray, weight=1.0
) -> None:
    next_state[0] = max(0, min(next_state[0], rows - 1))
    next_state[1] = max(0, min(next_state[1], cols - 1))
    next_idx = to_idx(next_state)
    mdp[state_idx, action_idx, next_idx] += weight


def init_mdp(num_states: int, num_actions: int, wind_col: list[int]) -> np.ndarray:
    mdp = np.zeros((num_states, num_actions, num_states))
    for idx in range(num_states):
        state = to_state(idx)
        if (state == t_state).all():
            add_transition(mdp, idx, num_actions - 1, state)
            continue

        for action_idx in get_valid_actions(state, idx=True):
            next_state = state + actions[action_idx]
            if wind_col[next_state[1]] != 0:
                wind = wind_col[next_state[1]]
                for noise in [-1, 0, 1]:
                    # Negative because the wind goes up
                    winded_state = next_state - np.array([noise + wind, 0])
                    add_transition(mdp, idx, action_idx, winded_state, weight=1 / 3)
            else:
                add_transition(mdp, idx, action_idx, next_state)
    return mdp


# Reward: One dimensional
# -1 for each step, 0 for the goal
def init_reward(
    num_states: int, t_state: np.ndarray = t_state, t_reward=0
) -> np.ndarray:
    reward = np.full((num_states, num_states), -1)
    reward[:, to_idx(t_state)] = t_reward
    reward[to_idx(t_state), to_idx(t_state)] = 0
    return reward
