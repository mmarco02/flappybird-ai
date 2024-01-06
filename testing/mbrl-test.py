from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf

from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.metrics import tf_metrics
from tf_agents.networks import categorical_q_network
from tf_agents.policies import random_tf_policy
from tf_agents.trajectories import trajectory, policy_step
from tf_agents.utils import common
from tf_env import FlappyBirdEnvironment
from flappybird import FlappyBird
import numpy as np
from tf_agents.policies import policy_saver
from gym import spaces

# Define constants
num_iterations = 15000
num_eval_episodes = 10
eval_interval = 1000

# Create an instance using gym.make
env = FlappyBirdEnvironment(FlappyBird())
train_py_env = env
eval_py_env = env

train_env = tf_py_environment.TFPyEnvironment(train_py_env)
eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

# Model-based agent
class MPCAgent:
    def __init__(self, environment):
        self.environment = environment
        self.observation_spec = environment.observation_spec()
        self.action_spec = environment.action_spec()
        self.action_space = spaces.Discrete(2)

    def plan(self, current_observation):
        return self.action_space.sample()

    def step(self, time_step):
        action = self.plan(time_step.observation)
        return policy_step.PolicyStep(action=action)

# Instantiate the MPC agent
mpc_agent = MPCAgent(train_env)

# Training loop
for _ in range(num_iterations):
    time_step = train_env.reset()
    episode_return = 0.0

    while not time_step.is_last():
        action_step = mpc_agent.step(time_step)
        next_time_step = train_env.step(action_step.action)
        traj = trajectory.from_transition(time_step, action_step, next_time_step)
        episode_return += time_step.reward

    avg_return = episode_return / num_eval_episodes

    if _ % eval_interval == 0:
        print('step = {0}: Average Return = {1:.2f}'.format(_, avg_return))
