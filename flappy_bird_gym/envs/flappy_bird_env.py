from typing import Optional, Union, List

import gym
from gym import spaces
import numpy as np

from flappybird import FlappyBird

class FlappyBirdEnvironment(gym.Env):
    def __init__(self):
        super(FlappyBirdEnvironment, self).__init__()
        self._flappy_bird_game = FlappyBird()
        self._action_space = spaces.Discrete(2)
        self._observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)
        self._state = None
        self._reward = 0.0
        self._episode_ended = False

    def reset(self):
        self._flappy_bird_game.reset()
        self._state = np.full((5,), 0.0, dtype=np.float32)
        self._reward = 0.0
        self._episode_ended = False
        return self._state, {}

    def step(self, action):
        if self._episode_ended:
            return self.reset()

        self._flappy_bird_game.render()

        observation = self._get_observation()

        if action == 1:
            self._flappy_bird_game.jump()

        self._reward = self._flappy_bird_game.score + (self._flappy_bird_game.distance / 100)

        if self._flappy_bird_game.game_over:
            self._episode_ended = True
            print(f"Episode ended with score {self._flappy_bird_game.score}"
                  f" and distance {self._flappy_bird_game.distance}")
            return observation, -10, self._episode_ended, {}

        return observation, self._reward, self._episode_ended, {}

    def _get_observation(self):
        bird_y = self._flappy_bird_game.get_bird_y()
        bird_velocity = self._flappy_bird_game.get_bird_velocity()
        distance_to_next_pipe = self._flappy_bird_game.get_distance_to_next_pipe()
        vertical_distance_to_next_bottom_pipe = self._flappy_bird_game.get_vertical_distance_to_next_bottom_pipe()
        vertical_distance_to_next_top_pipe = self._flappy_bird_game.get_vertical_distance_to_next_top_pipe()

        observation = np.array([
            bird_y,
            bird_velocity,
            distance_to_next_pipe,
            vertical_distance_to_next_top_pipe,
            vertical_distance_to_next_bottom_pipe,
        ], dtype=np.float32)

        return observation

    def random_action(self):
        return np.random.randint(0, 2)

    @property
    def action_space(self):
        return self._action_space

    @property
    def observation_space(self):
        return self._observation_space

    @classmethod
    def make(cls):
        return cls()
