from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
import os

from tf_agents.agents.categorical_dqn import categorical_dqn_agent
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import categorical_q_network
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common
from tf_env import FlappyBirdEnvironment
from flappybird import FlappyBird
import numpy as np
from tf_agents.policies import policy_saver
import pygame

num_iterations = 5000

initial_collect_steps = 15
collect_steps_per_iteration = 15
replay_buffer_capacity = 100000

fc_layer_params = (100,)

batch_size = 64
learning_rate = 1e-3
gamma = 0.99
log_interval = 200

num_atoms = 51
min_q_value = -20
max_q_value = 20
n_step_update = 2

num_eval_episodes = 10
eval_interval = 1000

# Create an instance using gym.make
env = FlappyBirdEnvironment(FlappyBird())

env.reset()

print('Observation Spec:')
print(env.time_step_spec().observation)
print('Action Spec:')
print(env.action_spec())

time_step = env.reset()
print('Time step:')
print(time_step)

action = np.array(1, dtype=np.int32)

next_time_step = env.step(action)
print('Next time step:')
print(next_time_step)

train_py_env = env
eval_py_env = env

train_env = tf_py_environment.TFPyEnvironment(train_py_env)
eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

categorical_q_net = categorical_q_network.CategoricalQNetwork(
    train_env.observation_spec(),
    train_env.action_spec(),
    num_atoms=num_atoms,
    fc_layer_params=fc_layer_params)

optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)

#train_step_counter = tf.compat.v2.Variable(0)

global_step = tf.compat.v1.train.get_or_create_global_step()

agent = categorical_dqn_agent.CategoricalDqnAgent(
    train_env.time_step_spec(),
    train_env.action_spec(),
    categorical_q_network=categorical_q_net,
    optimizer=optimizer,
    min_q_value=min_q_value,
    max_q_value=max_q_value,
    n_step_update=n_step_update,
    td_errors_loss_fn=common.element_wise_squared_loss,
    gamma=gamma,
    train_step_counter=global_step)

agent.initialize()


def compute_avg_return(environment, policy, num_episodes=10):
    total_return = 0.0
    for _ in range(num_episodes):

        time_step = environment.reset()
        episode_return = 0.0

        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = environment.step(action_step.action)
            episode_return += time_step.reward
        total_return += episode_return

    avg_return = total_return / num_episodes
    return avg_return.numpy()[0]


random_policy = random_tf_policy.RandomTFPolicy(train_env.time_step_spec(),
                                                train_env.action_spec())

compute_avg_return(eval_env, random_policy, num_eval_episodes)

# Please also see the metrics module for standard implementations of different
# metrics.

replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec=agent.collect_data_spec,
    batch_size=train_env.batch_size,
    max_length=replay_buffer_capacity)


def collect_step(environment, policy):
    time_step = environment.current_time_step()
    action_step = policy.action(time_step)
    next_time_step = environment.step(action_step.action)
    traj = trajectory.from_transition(time_step, action_step, next_time_step)

    # Add trajectory to the replay buffer
    replay_buffer.add_batch(traj)


for _ in range(initial_collect_steps):
    collect_step(train_env, random_policy)

# This loop is so common in RL, that we provide standard implementations of
# these. For more details see the drivers module.

# Dataset generates trajectories with shape [BxTx...] where
# T = n_step_update + 1.
dataset = replay_buffer.as_dataset(
    num_parallel_calls=3, sample_batch_size=batch_size,
    num_steps=n_step_update + 1).prefetch(3)

iterator = iter(dataset)

# (Optional) Optimize by wrapping some of the code in a graph using TF function.
agent.train = common.function(agent.train)

# Reset the train step
agent.train_step_counter.assign(0)

# Evaluate the agent's policy once before training.
avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
returns = [avg_return]

policy_dir = 'saved_policy'

# load the policy
#saved_policy = tf.compat.v2.saved_model.load(policy_dir)

checkpoint_dir = os.path.join(policy_dir, 'checkpoint')
train_checkpointer = common.Checkpointer(
    ckpt_dir=checkpoint_dir,
    max_to_keep=4,
    agent=agent,
    policy=agent.policy,
    collect_policy=agent.collect_policy,
    replay_buffer=replay_buffer,
    global_step=global_step
)

if train_checkpointer.checkpoint_exists:
    train_checkpointer.initialize_or_restore()
    global_step = tf.compat.v1.train.get_global_step()
    print(f"Checkpoint restored from {checkpoint_dir}")
else:
    print(f"Checkpoint not found in {checkpoint_dir}")

try:
    for _ in range(num_iterations):
        # Collect a few steps using collect_policy and save to the replay buffer.
        for _ in range(collect_steps_per_iteration):
            collect_step(train_env, agent.collect_policy)

        # Sample a batch of data from the buffer and update the agent's network.
        experience, unused_info = next(iterator)
        train_loss = agent.train(experience).loss

        step = agent.train_step_counter.numpy()

        if step % log_interval == 0:
            print('step = {0}: loss = {1}'.format(step, train_loss))
            train_checkpointer.save(global_step)
            print(f"Checkpoint saved to {checkpoint_dir}")

        if step % eval_interval == 0:
            avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
            print('step = {0}: Average Return = {1:.2f}'.format(step, avg_return))
            returns.append(avg_return)

    steps = range(0, num_iterations + 1, eval_interval)
    plt.plot(steps, returns)
    plt.ylabel('Average Return')
    plt.xlabel('Step')
    plt.show()

    train_checkpointer.save(global_step)
    print(f"Training ended, checkpoint saved to {checkpoint_dir}")

except KeyboardInterrupt:
    print("Interrupted")
    train_checkpointer.save(global_step)
    print(f"Checkpoint saved to {checkpoint_dir}")
    pygame.quit()
    sys.exit(0)
