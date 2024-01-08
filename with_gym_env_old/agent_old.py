# Import Required Libraries
import numpy as np
import tensorflow as tf
import gym
import matplotlib.pyplot as plt
import flappy_bird_gymnasium
import gymnasium

# Define the DQN Model
class DQN(tf.keras.Model):
    def __init__(self, num_actions):
        super(DQN, self).__init__()
        self.dense1 = tf.keras.layers.Dense(24, activation='relu')
        self.dense2 = tf.keras.layers.Dense(24, activation='relu')
        self.output_layer = tf.keras.layers.Dense(
            num_actions, activation='linear')

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return self.output_layer(x)


# CartPole has 2 possible actions: push left or push right
num_actions = 2
#if trained dqn model exists, load it
try:
    dqn_agent = DQN(num_actions)
    dqn_agent.load_weights("dqn_flappy_bird.keras")
    print("Loaded trained model")
except:
    dqn_agent = DQN(num_actions)
    print("No trained model found, training new model")

# Define the DQN Algorithm Parameters
learning_rate = 0.001
discount_factor = 0.99
# Initial exploration probability
exploration_prob = 1.0
# Decay rate of exploration probability
exploration_decay = 0.995
# Minimum exploration probability
min_exploration_prob = 0.1

log_interval = 50

env = gymnasium.make("FlappyBird-v0", render_mode="human")

# Define the Loss Function and Optimizer
loss_fn = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

# Training the DQN
num_episodes = 1000
max_steps_per_episode = 500

for episode in range(num_episodes):
    obs, _ = env.reset()
    episode_reward = 0

    for step in range(max_steps_per_episode):
        # Choose action using epsilon-greedy policy
        if np.random.rand() < exploration_prob:
            action = env.action_space.sample()  # Explore randomly
        else:
            if isinstance(obs, tuple):
                action = np.argmax(dqn_agent(obs[0][np.newaxis, :]))
            else:
                action = np.argmax(dqn_agent(obs[np.newaxis, :]))

        obs, reward, terminated, _, info = env.step(action)

        # Update the Q-values using Bellman equation
        with tf.GradientTape() as tape:
            if isinstance(obs, tuple):
                current_q_values = dqn_agent(obs[0][np.newaxis, :])
            else:
                current_q_values = dqn_agent(obs[np.newaxis, :])
            next_q_values = dqn_agent(obs[np.newaxis, :])
            max_next_q = tf.reduce_max(next_q_values, axis=-1)
            target_q_values = current_q_values.numpy()
            target_q_values[0, action] = reward + discount_factor * max_next_q * (1 - terminated)
            loss = loss_fn(current_q_values, target_q_values)

        gradients = tape.gradient(loss, dqn_agent.trainable_variables)
        optimizer.apply_gradients(zip(gradients, dqn_agent.trainable_variables))

        obs = obs
        episode_reward += reward

        if terminated:
            print(f"Episode {episode + 1}: Reward = {episode_reward}")
            break

    # Decay exploration probability
    exploration_prob = max(min_exploration_prob, exploration_prob * exploration_decay)
    if (episode + 1) % 100 == 0:
        print(f"Episode {episode + 1}: Reward = {episode_reward}")

# Evaluating the Trained DQN
num_eval_episodes = 10
eval_rewards = []

for _ in range(num_eval_episodes):
    obs, _ = env.reset()
    eval_reward = 0

    for _ in range(max_steps_per_episode):
        if isinstance(obs, tuple):
            action = np.argmax(dqn_agent(obs[0][np.newaxis, :]))
        else:
            action = np.argmax(dqn_agent(obs[np.newaxis, :]))

        obs, reward, terminated, _,  _ = env.step(action)
        eval_reward += reward
        obs = obs

        if terminated:
            break

    eval_rewards.append(eval_reward)

average_eval_reward = np.mean(eval_rewards)
print(f"Average Evaluation Reward: {average_eval_reward}")

# plot the rewards and iterations
plt.plot(range(num_eval_episodes), eval_rewards)
plt.xlabel("Episode")
plt.ylabel("Evaluation Reward")
plt.show()




