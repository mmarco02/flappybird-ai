import numpy as np
import gym
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from flappybird import FlappyBird
from tf_env import FlappyBirdEnvironment
import random

class QLearningAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = 0.95  # discount factor
        self.learning_rate = 0.001
        self.epsilon = 1.0  # exploration rate
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.memory = []  # memory to store (state, action, reward, next_state, done) tuples
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_size)
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])

    def replay(self, batch_size):
        batch = np.array(random.sample(self.memory, batch_size))
        for state, action, reward, next_state, done in batch:
            q_update = reward
            if not done:
                q_update = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            q_values = self.model.predict(state)
            q_values[0][action] = q_update
            self.model.fit(state, q_values, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# Environment
env = FlappyBirdEnvironment(FlappyBird())
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

# Create Q-learning agent
agent = QLearningAgent(state_size, action_size)

# Training parameters
episodes = 1000
batch_size = 32

# Training loop
for episode in range(episodes):
    state = env.reset()

    total_reward = 0
    for time in range(500):  # Change this if needed
        # env.render()  # Uncomment if you want to visualize the training

        action = agent.act(state)
        next_state = env.step(action)
        done = next_state.is_last()
        agent.remember(state, action, next_state.reward, next_state, done)
        state = next_state

        total_reward += state.reward

        if batch_size < len(agent.memory):
            agent.replay(batch_size)


        if done:
            print(f"Episode: {episode + 1}, Total Reward: {total_reward}")
            break

# Close the environment when done
env.close()
