import numpy as np
import random

random.seed(42)

class TicTacToeEnvironment:
    def __init__(self):
        self.grid = np.array(
            [
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0]
            ]
        )
        self.agent_move = 1
        self.human_move = 2
    def reset(self):
        self.grid = np.array(
            [
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0]
            ]
        )
        return self.grid
    def is_match(self, move):
        if self.grid[0, 0] == self.grid[0, 1] == self.grid[0, 2] == move:
            return True
        elif self.grid[1, 0] == self.grid[1, 1] == self.grid[1, 2] == move:
            return True
        elif self.grid[2, 0] == self.grid[2, 1] == self.grid[2, 2] == move:
            return True
        elif self.grid[0, 0] == self.grid[1, 0] == self.grid[2, 0] == move:
            return True
        elif self.grid[0, 1] == self.grid[1, 1] == self.grid[2, 1] == move:
            return True
        elif self.grid[0, 2] == self.grid[1, 2] == self.grid[2, 2] == move:
            return True
        elif self.grid[0, 0] == self.grid[1, 1] == self.grid[2, 2] == move:
            return True
        elif self.grid[0, 2] == self.grid[1, 1] == self.grid[2, 0] == move:
            return True
        else:
            return False
    def get_available_positions(self):
        available_positions = []
        for i in range(self.grid.shape[0]):
            for j in range(self.grid.shape[1]):
                if self.grid[i, j] == 0:
                    available_positions.append([i, j])
        return available_positions
    def step(self, action):
        if len(self.get_available_positions()) > 0:
            self.grid[action[0]][action[1]] = self.agent_move
            if self.is_match(self.agent_move):
                return 2, True, True
        if len(self.get_available_positions()) > 0:
            human_action = random.choice(self.get_available_positions())
            self.grid[human_action[0]][human_action[1]] = self.human_move
            if self.is_match(self.human_move):
                return -1, True, False
        if len(self.get_available_positions()) == 0:
            return 1, True, False
        return 0, False, False
    def render(self):
        print(self.grid)

def q_learning(env, alpha=0.001, gamma=0.95, epsilon=0.1, episodes=10000):
    q_table = {}
    agent_wins = 0
    for _ in range(episodes):
        state = tuple(map(tuple, env.reset()))
        done = False
        while not done:
            if random.uniform(0, 1) < epsilon:
                action = random.choice(env.get_available_positions())
            else:
                action = max(env.get_available_positions(), key=lambda a: q_table.get((state, tuple(a)), 0))
            reward, done, agent_win = env.step(action)
            next_state = tuple(map(tuple, env.grid))
            old_value = q_table.get((state, tuple(action)), 0)
            future_max = max([q_table.get((next_state, tuple(a)), 0) for a in env.get_available_positions()], default=0)
            q_table[(state, tuple(action))] = old_value + alpha * (reward + gamma * future_max - old_value)
            state = next_state
            if agent_win:
                agent_wins += 1
    return q_table, agent_wins


def play_with_trained_model(env, q_table):
    env.reset()
    env.render()

    while True:
        # Human Move
        human_move = list(map(int, input("Enter your move (row and column): ").split()))
        if human_move not in env.get_available_positions():
            print("Invalid move. Try again.")
            continue
        env.grid[human_move[0]][human_move[1]] = env.human_move
        env.render()
        if env.is_match(env.human_move):
            print("You win!")
            break
        if not env.get_available_positions():
            print("It's a draw!")
            break

        # AI Move with Winning & Blocking Strategy
        state = tuple(map(tuple, env.grid))
        best_move = None
        best_q_value = float('-inf')

        for move in env.get_available_positions():
            # Check if AI can win in the next move
            env.grid[move[0]][move[1]] = env.agent_move
            if env.is_match(env.agent_move):
                print("AI wins!")
                env.render()
                return
            env.grid[move[0]][move[1]] = 0  # Undo move

            # Check if AI can block human from winning
            env.grid[move[0]][move[1]] = env.human_move
            if env.is_match(env.human_move):
                best_move = move  # Block human
            env.grid[move[0]][move[1]] = 0  # Undo move

            # Choose move with the highest Q-value if no immediate win or block
            q_value = q_table.get((state, tuple(move)), 0)
            if q_value > best_q_value:
                best_q_value = q_value
                best_move = move

        # Perform AI move
        env.grid[best_move[0]][best_move[1]] = env.agent_move
        print("AI's move:")
        env.render()
        if env.is_match(env.agent_move):
            print("AI wins!")
            break
        if not env.get_available_positions():
            print("It's a draw!")
            break

if __name__ == '__main__':
    env = TicTacToeEnvironment()
    q_table, agent_wins = q_learning(env)
    print(f'Agent Win Percentage: {agent_wins}/10000 ({agent_wins/10000*100:.2f}%)')
    play_with_trained_model(env, q_table)

