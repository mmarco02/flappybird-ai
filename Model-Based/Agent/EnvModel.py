from keras.models import Sequential
from keras.layers import Dense, Input
import numpy as np
import tensorflow as tf
import os

class EnvironmentModel:
    def __init__(self, state_size):
        self.model = Sequential([
            Input(shape=(6,)),
            Dense(128, activation='relu'),
            Dense(128, activation='relu'),
            Dense(state_size + 1)  # Predicting next state and reward
        ])
        self.model.compile(optimizer='adam', loss='mse')

    def predict(self, state, action):
        action = action.reshape(-1, 1)  # Reshape action to 2D
        input_data = np.concatenate([state, action], axis=-1)
        prediction = self.model.predict(input_data, verbose=0)
        predicted_next_state = prediction[:, :-1]  # All but the last element
        predicted_reward = prediction[:, -1:]  # Only the last element
        return predicted_next_state, predicted_reward

    def train(self, states, actions, next_states, rewards):
        actions = actions.reshape(-1, 1)  # Reshape to 2D if necessary
        rewards = rewards.reshape(-1, 1)  # Reshape to 2D if necessary

        inputs = np.concatenate([states, actions], axis=-1)
        outputs = np.concatenate([next_states, rewards], axis=-1)

        # print("State shape:", np.array([state]).shape)
        # print("Action shape:", np.array([action]).reshape(-1, 1).shape)
        # print("Next state shape:", np.array([next_state]).shape)
        # print("Reward shape:", np.array([reward]).reshape(-1, 1).shape)

        # print("Inputs shape:", inputs.shape)
        # print("Outputs shape:", outputs.shape)

        self.model.fit(inputs, outputs, epochs=10, verbose=0)

    def save_model(self, directory="saved"):
        if not os.path.exists(directory):
            os.makedirs(directory)
        model_path = os.path.join(directory, 'env_model.keras')
        self.model.save(model_path)
        print("Environment model saved.")

    def load_model(self, directory="saved"):
        model_path = os.path.join(directory, 'env_model.keras')
        if os.path.exists(model_path):
            self.model = tf.keras.models.load_model(model_path)
            print("Environment model loaded.")
        else:
            print("Saved environment model not found. Starting from scratch.")
