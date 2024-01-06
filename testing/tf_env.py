import numpy as np
from tf_agents.environments import tf_py_environment
from tf_agents.trajectories import time_step as ts
from tf_agents.specs import array_spec
from flappybird import FlappyBird


class FlappyBirdEnvironment(tf_py_environment.py_environment.PyEnvironment):
    def __init__(self, fbgame):
        super().__init__()
        self._observation_shape = (5,)
        self._flappy_bird_game = fbgame
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=1, name='action')
        self._observation_spec = array_spec.ArraySpec(
            shape=self._observation_shape, dtype=np.float32, name='observation')
        self.action_space = np.array([0, 1])
        self._reward = 0.0
        self._score = 0
        self._distance = 0
        self._episode_ended = False
        self._observation = np.full(self._observation_shape, 0, dtype=np.float32)

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        self._episode_ended = False
        self._flappy_bird_game.reset()
        self._observation = np.full(self._observation_shape, 0, dtype=np.float32)
        self._reward = 0.0
        return ts.restart(self._observation)

    def random_action(self):
        self.random_action = np.random.Generator(np.random.MT19937()).integers(0, 2)
        return self.random_action

    def _step(self, action):
        if self._episode_ended:
            return self.reset()

        if action.item() == 1:
            self._flappy_bird_game.jump()

        observation = self.get_observation()
        self.set_observation(observation)

        self._flappy_bird_game.render()

        self._score = self._flappy_bird_game.get_score()
        self._distance = self._flappy_bird_game.get_distance()

        self._reward = self._score + (self._distance / 100)

        if self._flappy_bird_game.get_collided():
            self._episode_ended = True
            print(f"Episode ended, reward: {self._reward}, score: {self._score}, distance: {self._distance}")
            return ts.termination(observation=observation, reward=-5.0)

        return ts.transition(observation=observation, reward=self._reward, discount=1.0)

    def set_observation(self, observation):
        self._observation = observation

    def get_observation(self):
        bird_y = self._flappy_bird_game.get_bird_y()
        bird_velocity = self._flappy_bird_game.get_bird_velocity()
        distance_to_next_pipe = self._flappy_bird_game.get_distance_to_next_pipe()
        vertical_distance_to_next_top_pipe = self._flappy_bird_game.get_vertical_distance_to_next_top_pipe()
        vertical_distance_to_next_bottom_pipe = self._flappy_bird_game.get_vertical_distance_to_next_bottom_pipe()

        observation = np.array([
            bird_y,
            bird_velocity,
            distance_to_next_pipe,
            vertical_distance_to_next_top_pipe,
            vertical_distance_to_next_bottom_pipe
        ], dtype=np.float32)

        return observation

    def get_score(self):
        return self._score

    def get_distance(self):
        return self._distance

    def get_reward(self):
        return self._reward

    def get_episode_ended(self):
        return self._episode_ended