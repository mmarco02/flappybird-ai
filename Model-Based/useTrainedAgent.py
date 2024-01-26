import numpy as np
import gym
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

class DQN():
    def __init__(self):
        self.learning_rate = 0.001

        self.model = self.DQNmodel()

    def DQNmodel(self):
        model = Sequential()
        model.add(Dense(32, input_shape=(5,), activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(2, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=self.learning_rate), metrics=['accuracy'])
        return model

    def act(self, state):
        q_values = self.model.predict(state.reshape(1, -1))
        return np.argmax(q_values)  # Choose the best action

gym.register(id='FlappyBird-m', entry_point='flappy_bird_gym.envs.flappy_bird_env:FlappyBirdEnvironment')

# Initialize Gym environment
env = gym.make('FlappyBird-m')

# Initialize agent
agent = DQN()

# Load the model
try:
    agent.model.load_weights("my_model.keras")
    print("Loaded saved model.")
except:
    print("No saved model found. Cannot play the game without a trained model.")
    exit()

num_episodes = 100  # Number of episodes you want to play

for episode in range(num_episodes):
    state = env.reset()
    total_reward = 0
    done = False

    while not done:
        action = agent.act(state)  # Select action

        # Take action and observe the next state and reward
        next_state, reward, done, _ = env.step(action)
        total_reward += reward

        state = next_state

    print(f"Episode: {episode}, Total reward: {total_reward}")

env.close()
