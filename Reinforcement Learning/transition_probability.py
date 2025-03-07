import numpy as np

labels = {
    0: ['Study', 1],
    1: ['Youtube', -1],
    2: ['Eat', 0],
    3: ['Sleep', -1],
    4: ['Go Home', 0]
}

P = np.array([
    [0.6, 0.4, 0.0, 0.0, 0.0],
    [0.4, 0.0, 0.6, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.6, 0.4],
    [0.0, 0.0, 0.0, 0.0, 1.0],
    [0.0, 0.0, 0.0, 0.0, 1.0],
])

def simulate_markov_process(initial_state):
    state = initial_state
    sequence = [labels[state][0]]
    rewards = []
    max_reward = 0
    while state != 4 and max_reward <= 1:
        state = np.random.choice(5, p=P[state])
        sequence.append(labels[state][0])
        rewards.append(labels[state][1])
        max_reward = max(max_reward, sum(rewards))
    return sequence, max_reward

initial_state = 0
result = simulate_markov_process(initial_state)
print(f'Best Sequence: {result[0]}, Reward: {result[1]}')