import numpy as np

class TabularQLearning:
    def __init__(self, env, eval_env):
        self.env = env
        self.eval_env = eval_env

        self.n_states = int(self.env.observation_space.n)
        self.n_actions = int(self.env.action_space.n)
        self.q_table = np.zeros((self.n_states, self.n_actions))

    def evaluate(self, n_eval_episodes=500, gamma=0.99):
        eval_returns = []
        for _ in range(n_eval_episodes):
            state, _ = self.eval_env.reset()
            terminated = False
            truncated = False
            episode_return = 0.0
            step_count = 0

            while not (terminated or truncated):
                # Completely greedy policy
                action = int(np.argmax(self.q_table[state]))
                next_state, reward, terminated, truncated, _ = self.eval_env.step(action)

                # Discounted return
                episode_return += reward * (gamma ** step_count)
                state = next_state
                step_count += 1

            eval_returns.append(episode_return)

        return np.mean(eval_returns)

    def train(self, total_training_steps=500_000, gamma=0.99, alpha=0.1,
              epsilon_start=1.0, epsilon_end=0.05, epsilon_decay_steps=300_000):

        evaluations = []
        state, _ = self.env.reset()

        for step in range(total_training_steps):
            # Evaluate every 10_000 training steps
            if step % 10_000 == 0:
                mean_return = self.evaluate(n_eval_episodes=500, gamma=gamma)
                evaluations.append((step, mean_return))

            # Linearly decay epsilon
            epsilon = epsilon_start - (step / epsilon_decay_steps) * (epsilon_start - epsilon_end)
            epsilon = max(epsilon_end, epsilon)

            # Epsilon-greedy action selection
            if np.random.rand() < epsilon:
                action = self.env.action_space.sample()
            else:
                action = int(np.argmax(self.q_table[state]))

            next_state, reward, terminated, truncated, _ = self.env.step(action)

            # Q-learning TD update
            best_next_action = int(np.argmax(self.q_table[next_state]))
            td_target = reward + gamma * self.q_table[next_state, best_next_action] * (not terminated)
            td_error = td_target - self.q_table[state, action]
            self.q_table[state, action] += alpha * td_error

            if terminated or truncated:
                state, _ = self.env.reset()
            else:
                state = next_state

        # Final evaluation at the very end
        mean_return = self.evaluate(n_eval_episodes=500, gamma=gamma)
        evaluations.append((total_training_steps, mean_return))

        return evaluations
