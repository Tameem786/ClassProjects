import random
import numpy as np

class Agent:
    def __init__(self):
        self.q_table = np.zeros((5, 7, 4))
        self.returns_sum = np.zeros((5, 7, 4))
        self.returns_count = np.zeros((5, 7, 4))
        self.eps = 0.9
    def select_action(self, s):
        x, y = s
        if random.random() < self.eps:
            return random.randint(0, 3)
        else:
            return np.argmax(self.q_table[x][y])
    def update_q_table(self, episode):
        G = 0
        visited = set()
        for transition in reversed(episode):
            s, a, r = transition
            x, y = s
            G = r + G
            if (x, y, a) not in visited:
                self.returns_sum[x, y, a] += G
                self.returns_count[x, y, a] += 1
                self.q_table[x, y, a] += self.returns_sum[x, y, a] / self.returns_count[x, y, a]
                visited.add((x, y, a))
    def anneal_eps(self):
        self.eps = max(self.eps - 0.01, 0.1)

class GridWorld:
    def __init__(self):
        self.x = 0
        self.y = 0
    def step(self, a):
        if a == 0:
            self.move_left()
        elif a == 1:
            self.move_up()
        elif a == 2:
            self.move_right()
        elif a == 3:
            self.move_down()
        reward = -1
        done = self.is_done()
        if self.x == 1 and self.y == 1:
            reward = -5
        elif self.x == 3 and self.y == 3:
            reward = 10
        return (self.x, self.y), reward, done
    def move_right(self):
        if self.y < 6:
            self.y += 1
    def move_left(self):
        if self.y > 0:
            self.y -= 1
    def move_down(self):
        if self.x < 4:
            self.x += 1
    def move_up(self):
        if self.x > 0:
            self.x -= 1
    def is_done(self):
        return self.x == 3 and self.y == 3
    def reset(self):
        self.x = 0
        self.y = 0
        return self.x, self.y

def main():
    env = GridWorld()
    agent = Agent()

    for episode_count in range(1000):
        done = False
        episode = []
        s = env.reset()
        while not done:
            a = agent.select_action(s)
            s, r, done = env.step(a)
            episode.append((s, a, r))
        agent.update_q_table(episode)
        agent.anneal_eps()
    for row in agent.q_table:
        print(row)

if __name__ == '__main__':
    main()