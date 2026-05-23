import numpy as np

class TabularREINFORCE:
    def __init__(self, env, eval_env):
        self.env = env
        self.eval_env = eval_env

        self.n_states = int(self.env.observation_space.n)
        self.n_actions = int(self.env.action_space.n)

        # Policy parameter matrix
        self.theta = np.zeros((self.n_states, self.n_actions))

        # State-value array for the baseline
        self.V = np.zeros(self.n_states)

    def softmax(self, state):
        x = self.theta[state]
        # Numerically stable softmax
        x = x - np.max(x)
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x)
    def policy_probs(self, state):
        return self.softmax(state)
    def evaluate(self, n_eval_episodes=500, gamma=0.99):
        eval_returns = []
        for _ in range(n_eval_episodes):
            state, _ = self.eval_env.reset()
            terminated = False
            truncated = False
            episode_return = 0.0
            step_count = 0

            while not (terminated or truncated):
                # Strictly greedy policy (argmax of probs)
                probs = self.policy_probs(state)
                action = int(np.argmax(probs))

                next_state, reward, terminated, truncated, _ = self.eval_env.step(action)

                # Discounted return calculation
                episode_return += reward * (gamma ** step_count)
                state = next_state
                step_count += 1

            eval_returns.append(episode_return)

        return float(np.mean(eval_returns))

    def train(self, total_training_steps=500_000, gamma=0.99, policy_lr=0.005, value_lr=0.05,
              theta_clip=20.0, use_baseline=False):

        evaluations = []
        state, _ = self.env.reset()
        states = []
        actions = []
        rewards = []

        for step in range(total_training_steps):
            # Evaluate every 10_000 training steps
            if step % 10_000 == 0:
                mean_return = self.evaluate(n_eval_episodes=500, gamma=gamma)
                evaluations.append((step, mean_return))

            probs = self.policy_probs(state)
            action = np.random.choice(self.n_actions, p=probs)
            next_state, reward, terminated, truncated, _ = self.env.step(action)

            states.append(state)
            actions.append(action)
            rewards.append(reward)

            if terminated or truncated:
                # Calculate G_t for each step in the trajectory
                G = np.zeros(len(rewards))
                g_t = 0.0
                for t in reversed(range(len(rewards))):
                    g_t = rewards[t] + gamma * g_t
                    G[t] = g_t

                # Perform the updates
                for t in range(len(states)):
                    s_t = states[t]
                    a_t = actions[t]
                    g_t = G[t]

                    if use_baseline:
                        delta = g_t - self.V[s_t]
                        self.V[s_t] = self.V[s_t] + value_lr * delta
                    else:
                        delta = g_t

                    # Update the policy weights
                    step_probs = self.softmax(s_t)
                    grad = -step_probs.copy()
                    grad[a_t] += 1.0  # gradient is 1 - prob(a_t) for chosen action, -prob(a) for others

                    self.theta[s_t] += policy_lr * (gamma ** t) * delta * grad

                # Clip the entire theta matrix to be within [-theta_clip, theta_clip]
                self.theta = np.clip(self.theta, -theta_clip, theta_clip)

                # Reset for the next episode
                states = []
                actions = []
                rewards = []
                state, _ = self.env.reset()
            else:
                state = next_state

        # Final evaluation at the very end
        mean_return = self.evaluate(n_eval_episodes=500, gamma=gamma)
        evaluations.append((total_training_steps, mean_return))

        return evaluations
