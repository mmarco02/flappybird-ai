# Import Required Libraries
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import gym
from flappy_bird_gym.envs.flappy_bird_env import FlappyBirdEnvironment
from flappybird import FlappyBird
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
if tf.io.gfile.exists("dqn_flappy_bird.keras"):
    dqn_agent = DQN(num_actions)
    dqn_agent.load_weights("dqn_flappy_bird.keras")
    print("Trained model loaded")
else:
    dqn_agent = DQN(num_actions)
    print("New model created")

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

# Register the environment
gym.register(id='FlappyBird-m', entry_point='flappy_bird_gym.envs.flappy_bird_env:FlappyBirdEnvironment')

# Create an instance using gym.make
env = gym.make('FlappyBird-m')

# Define the Loss Function and Optimizer
loss_fn = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

# Training the DQN
num_episodes = 50
max_steps_per_episode = 500


for episode in range(num_episodes):
    obs, _ = env.reset()
    episode_reward = 0

    for step in range(max_steps_per_episode):
        # Choose action using epsilon-greedy policy
        if np.random.Generator(np.random.PCG64()).uniform(0, 1) < exploration_prob:
            action = env.action_space.sample()
        else:
            if isinstance(obs, tuple):
                action = np.argmax(dqn_agent(obs[0][np.newaxis, :]))
            else:
                action = np.argmax(dqn_agent(obs[np.newaxis, :]))

        obs, reward, terminated, info = env.step(action)
        #print(f"Observation: {obs} at step {step}")

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
            env.reset()
            break

    # Decay exploration probability
    exploration_prob = max(min_exploration_prob, exploration_prob * exploration_decay)
    if (episode + 1) % 100 == 0:
        print(f"Episode {episode + 1}: Reward = {episode_reward}")


# Evaluating the Trained DQN
num_eval_episodes = 10

print(f"Training finished. Evaluating the trained DQN for {num_eval_episodes} episodes.")

eval_rewards = []

for _ in range(num_eval_episodes):
    obs, _ = env.reset()
    eval_reward = 0

    for _ in range(max_steps_per_episode):
        if isinstance(obs, tuple):
            action = np.argmax(dqn_agent(obs[0][np.newaxis, :]))
        else:
            action = np.argmax(dqn_agent(obs[np.newaxis, :]))

        obs, reward, terminated, info = env.step(action)
        eval_reward += reward
        obs = obs

        if terminated:
            env.reset()
            print(f"Evaluation Reward = {eval_reward}")
            break

    eval_rewards.append(eval_reward)

average_eval_reward = np.mean(eval_rewards)
print(f"Average Evaluation Reward: {average_eval_reward}")

# plot the rewards and iterations
plt.plot(range(num_eval_episodes), eval_rewards)
plt.xlabel("Episode")
plt.ylabel("Evaluation Reward")
plt.show()

# Save the Trained DQN
dqn_agent.save_weights("dqn_flappy_bird.keras")





