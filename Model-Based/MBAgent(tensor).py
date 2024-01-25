from keras import Sequential
from keras.optimizers import Adam
from keras.layers import Dense
from collections import deque
import numpy as np
from matplotlib import pyplot as plt
import gym
import random
import tensorflow as tf

gym.register(id='FlappyBird-m', entry_point='flappy_bird_gym.envs.flappy_bird_env:FlappyBirdEnvironment')

env = gym.make('FlappyBird-m')

class DQN():
    def __init__(self):
        self.discount_factor = 0.99
        self.learning_rate = 0.001
        self.eps_min = 0.1
        self.eps_max = 1.0
        self.eps_decay_steps = 2000000
        self.eps_decay_rate = (self.eps_max - self.eps_min) / self.eps_decay_steps
        self.replay_memory_size = 5000
        self.replay_memory = deque([], maxlen=self.replay_memory_size)
        self.n_steps = 40000  # total number of training steps
        self.training_start = 5  # start training after 10,000 game iterations
        self.training_interval = 4  # run a training step every 4 game iterations
        self.save_steps = 50  # save the model every 1,000 training steps
        self.copy_steps = 100  # copy online DQN to target DQN every 10,000 training steps
        self.discount_rate = 0.99
        self.batch_size = 64
        self.iteration = 0  # game iterations
        self.done = True  # env needs to be reset
        self.rewards_history = []

        self.model = self.DQNmodel()
        self.target_model = self.DQNmodel()
        self.update_target_model()  # Initialize target model with same weights

        return

    def DQNmodel(self):
        model = Sequential()
        model.add(Dense(32, input_shape=(5,), activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(2, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=self.learning_rate), metrics=['accuracy'])
        return model

    def sample_memories(self, batch_size):
        # Ensure there are enough samples
        batch_size = min(batch_size, len(self.replay_memory))

        indices = np.random.permutation(len(self.replay_memory))[:batch_size]
        cols = [[], [], [], [], []]  # state, action, reward, next_state, continue
        for idx in indices:
            memory = self.replay_memory[idx]
            for col, value in zip(cols, memory):
                col.append(value)
        cols = [np.array(col) for col in cols]
        return cols[0], cols[1], cols[2].reshape(-1, 1), cols[3], cols[4].reshape(-1, 1)

    def epsilon_greedy(self, q_values, step):
        self.epsilon = max(self.eps_min, self.eps_max - (self.eps_max - self.eps_min) * step / self.eps_decay_steps)
        if np.random.rand() < self.epsilon:
            return np.random.randint(2)  # random action
        else:
            return np.argmax(q_values)  # optimal action

    def update_target_model(self):
        """Copy weights from the online model to the target model."""
        self.target_model.set_weights(self.model.get_weights())

    def train_model(self):
        if len(self.replay_memory) < self.batch_size:
            return
        mini_batch = random.sample(self.replay_memory, self.batch_size)

        states = np.array([sample[0] for sample in mini_batch])
        actions = np.array([sample[1] for sample in mini_batch])
        rewards = np.array([sample[2] for sample in mini_batch])
        next_states = np.array([sample[3] for sample in mini_batch])
        dones = np.array([sample[4] for sample in mini_batch])

        model_params = self.model.trainable_variables
        with tf.GradientTape() as tape:
            predicts = self.model(states)
            one_hot_action = tf.one_hot(actions, self.get_action_space())
            predicts = tf.reduce_sum(one_hot_action * predicts, axis=1)

            target_predicts = self.model(next_states)
            target_value = rewards + (1 - dones) * self.discount_factor * np.amax(target_predicts)

            loss = tf.reduce_mean(tf.square(target_value - predicts))

        grads = tape.gradient(loss, model_params)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.optimizer.apply_gradients(zip(grads, model_params))

    def save_model(self, episode):
        if episode % self.save_steps == 0:
            self.model.save('my_model.keras')
            self.target_model.save('my_target_model.keras')
            print("Model saved.")


    def epsilon_decay(self, episode):
        self.epsilon = self.eps_min + (self.eps_max - self.eps_min) * np.exp(-self.eps_decay_rate * episode)

    def get_action_space(self):
        return env.action_space.n


agent = DQN()

try:
    agent.model.load_weights("my_model.keras")
    agent.target_model.load_weights("my_target_model.keras")
    print("Loaded saved model.")
except:
    print("No saved model found. Creating new model.")

try:
    for episode in range(agent.n_steps):
        state = env.reset()
        total_reward = 0
        done = False
        steps = 0

        while not done:
            # Epsilon-greedy policy for action selection
            q_values = agent.model.predict(state.reshape(1, -1))
            action = agent.epsilon_greedy(q_values, agent.iteration)

            # Take action and observe the next state and reward
            next_state, reward, done, _ = env.step(action)
            total_reward += reward

            # Store transition in replay memory
            agent.replay_memory.append((state, action, reward, next_state, 1.0 - done))
            state = next_state

            # Train the model
            agent.train_model()

            steps += 1
        if episode % agent.copy_steps == 0:
            print("Copied model parameters to target network.")
            agent.update_target_model()

        if episode % agent.save_steps == 0:
            agent.save_model(episode)

            agent.iteration += 1

            agent.rewards_history.append(total_reward)

        print(f"Episode: {episode}, Total reward: {total_reward}, Epsilon: {agent.epsilon}")

except KeyboardInterrupt:
    agent.model.save("my_model.keras")
    agent.target_model.save("my_target_model.keras")
    print("Training interrupted, model saved.")

agent.model.save("my_model.keras")

# Plot the training progress
plt.figure(figsize=(8, 4))
plt.plot(agent.iteration, agent.rewards_history)
plt.xlabel("Episode", fontsize=14)
plt.ylabel("Training reward", fontsize=14)
plt.show()

# plot the agents accuracy
plt.figure(figsize=(8, 4))
plt.plot(agent.iteration, agent.epsilon)
plt.xlabel("Episode", fontsize=14)
plt.ylabel("Epsilon", fontsize=14)
plt.show()

