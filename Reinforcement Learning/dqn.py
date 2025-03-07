import gymnasium as gym
import random
import torch
import torch.nn as nn
import numpy as np
from collections import deque

learning_rate = 0.001
gamma = 0.99
epsilon_start = 1.0
epsilon_end = 0.01
epsilon_decay = 500
batch_size = 64
target_update_freq = 1000
memory_size = 10000

class QNetwork(nn.Module):
    def __init__(self):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(4, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 2)
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    def add(self, transition):
        self.buffer.append(transition)
    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*transitions)
        return np.vstack(states), actions, rewards, np.vstack(next_states), dones
    def size(self):
        return len(self.buffer)

def select_action(state, q_network, epsilon):
    if random.random() < epsilon:
        return random.randint(0, 1)
    else:
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = q_network(state_tensor)
            return q_values.argmax().item()

def train_dqn(env, q_network, target_network, optimizer, memory, episodes=1000):
    epsilon = epsilon_start
    epsilon_decay_step = (epsilon_start - epsilon_end) / epsilon_decay
    for episode in range(episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0
        while not done:
            action = select_action(state, q_network, epsilon)
            next_state, reward, done, _, _ = env.step(action)
            memory.add((state, action, reward, next_state, done))
            state = next_state
            total_reward += reward
            if memory.size() > batch_size:
                states, actions, rewards, next_states, dones = memory.sample(batch_size)
                states = torch.FloatTensor(states)
                next_states = torch.FloatTensor(next_states)
                rewards = torch.FloatTensor(rewards).unsqueeze(1)
                dones = torch.FloatTensor(dones).unsqueeze(1)
                q_values = q_network(states).gather(1, torch.tensor(actions).unsqueeze(1))
                next_q_values = target_network(next_states).max(1)[0].unsqueeze(1)
                target_q_values = rewards + gamma * next_q_values * (1-dones)
                loss = nn.functional.mse_loss(q_values, target_q_values)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        epsilon = max(epsilon_end, epsilon - epsilon_decay_step)
        if episode % target_update_freq == 0:
            target_network.load_state_dict(q_network.state_dict())
        if episode % 100 == 0:
            print(f'Episode {episode}, Total Reward: {total_reward}')

if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    q_network = QNetwork()
    target_network = QNetwork()
    target_network.load_state_dict(q_network.state_dict())
    optimizer = torch.optim.Adam(q_network.parameters(), lr=learning_rate)
    memory = ReplayBuffer(memory_size)
    train_dqn(env, q_network, target_network, optimizer, memory)
    env.close()