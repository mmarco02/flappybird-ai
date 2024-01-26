import numpy as np
import gym
from matplotlib import pyplot as plt
from DQN import DQN
from EnvModel import EnvironmentModel

gym.register(id='FlappyBird-m', entry_point='flappy_bird_gym.envs.flappy_bird_env:FlappyBirdEnvironment')

env = gym.make('FlappyBird-m')

agent = DQN()

state_size = env.observation_space.shape[0]
action_size = env.action_space.n  # This should be 1 if it's a discrete action space

env_model = EnvironmentModel(state_size, action_size)

total_episodes = 1000

save_interval = 50

try:
    agent.load_model()
    env_model.load_model()
except:
    print("Models not found. Starting from scratch.")

try:
    for episode in range(total_episodes):
        state = env.reset()
        done = False
        while not done:
            q_values = agent.model.predict(state.reshape(1, -1))
            action = np.argmax(q_values[0])
            next_state, reward, done, _ = env.step(action)

            # Train environment model
            env_model.train(np.array([state]), np.array([action]), np.array([next_state]), np.array([reward]))

            # Use environment model for planning
            predicted_next_state, predicted_reward = env_model.predict(np.array([state]), np.array([action]))

            # Update DQN with the predicted outcomes
            agent.update(state, action, predicted_reward, predicted_next_state)
            state = next_state

            if done:
                agent.rewards_history.append(reward)
                print(
                    "Episode: {}/{}, reward: {}".format(episode, total_episodes, reward))
                break

        if episode % save_interval == 0:
            agent.save_model()
            env_model.save_model()

except KeyboardInterrupt:
    agent.save_model()
    env_model.save_model()
    print("Training stopped. Models saved.")

rewards = agent.rewards_history
episodes = range(len(rewards))

plt.plot(episodes, rewards)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Sample Efficiency: Reward per Episode')
plt.show()