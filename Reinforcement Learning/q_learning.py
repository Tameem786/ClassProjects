"""
Q-Learing Equation
Q(s,a) = Q(s,a) + alpha[R + gamma*maxQ(s',a') - Q(s,a)]

+ - -+ - -+ - -+ - -+
| 0 | 0 | 0 | G | -> Goal = 1 (Reward +10)
+ - -+ - -+ - -+ - -+
| 0 | X | 0 | 0 | -> Wall = X (Reward -1)
+ - -+ - -+ - -+ - -+
| 0 | 0 | 0 | 0 |
+ - -+ - -+ - -+ - -+
| S | 0 | 0 | 0 | -> Start = (3, 0)
+ - -+ - -+ - -+ - -+
"""
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
import time

from sympy.physics.units import action

alpha = 0.1
gamma = 0.9
epsilon = 0.2
episodes = 500

class GridEnvironment:
    def __init__(self):
        self.grid = np.array([
            [0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0,-1, 0],
            [0, 0, 0, 0,-1, 0],
            [0, 0, 0, 0,-1, 0],
            [0, 0, 0, 0,-1, 0],
            [0, 0, 0, 0, 0, 0],
        ])
        self.start_pos = (5, 0)
        self.agent_pos = self.start_pos
    def reset(self):
        self.agent_pos = self.start_pos
        return self.agent_pos
    def step(self, action):
        x, y = self.agent_pos
        if action == 0 and x > 0:
            x -= 1
        elif action == 1 and y < 5:
            y += 1
        elif action == 2 and x < 5:
            x += 1
        elif action == 3 and y > 0:
            y -= 1

        if self.grid[x, y] == -1:
            reward = -10
            done = True
        elif self.grid[x, y] == 1:
            reward = 10
            done = True
        else:
            reward = -1
            done = False

        self.agent_pos = (x, y)
        return self.agent_pos, reward, done
    def render(self):
        env = self.grid.copy()
        x, y = self.agent_pos
        env[x, y] = 8
        print(env)

if __name__ == '__main__':
    env = GridEnvironment()
    Q_table = np.zeros((6, 6, 4))
    plt.ion()
    fig, ax = plt.subplots()

    for episode in range(episodes):
        state = env.reset()
        done = False
        while not done:
            x, y = state
            if random.uniform(0, 1) < epsilon:
                action = random.randint(0, 3)
            else:
                action = np.argmax(Q_table[x, y])
            next_state, reward, done = env.step(action)
            nx, ny = next_state
            old_value = Q_table[x, y, action]
            next_max = np.max(Q_table[nx, ny])
            new_value = (1-alpha)*old_value + alpha * (reward + gamma * next_max)
            Q_table[x, y, action] = new_value
            state = next_state

            ax.clear()
            sns.heatmap(np.max(Q_table, axis=2), annot=True, fmt='.2f', cmap='coolwarm', ax=ax)
            ax.set_title(f'Q-Table Episode: {episode+1}')
            plt.pause(0.01)
    plt.ioff()
    plt.show()

    # state = env.reset()
    # done = False
    # path = [state]
    # while not done:
    #     x, y = state
    #     action = np.argmax(Q_table[x, y])
    #     next_state, reward, done = env.step(action)
    #     path.append(next_state)
    #     state = next_state
    #     env.render()
    # print(f'Optimal Path: {path}')