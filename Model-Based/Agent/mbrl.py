import numpy as np
from matplotlib import pyplot as plt
from DQN import DQN
from EnvModel import EnvironmentModel
from Env import env

agent = DQN()

state_size = env.observation_space.shape[0]

env_model = EnvironmentModel(state_size)

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
            q_values = agent.model.predict(state.reshape(1, -1), verbose=0)
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
    losses = agent.loss_history
    episodes = range(len(rewards))

    plt.plot(range(len(agent.rewards_history)), agent.rewards_history)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Reward per Episode')
    plt.show()

    plt.plot(range(len(agent.loss_history)), agent.loss_history)
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title('Loss per Episode')
    plt.show()

rewards = agent.rewards_history
losses = agent.loss_history
episodes = range(len(rewards))

plt.plot(range(len(agent.rewards_history)), agent.rewards_history)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Reward per Episode')
plt.show()

plt.plot(range(len(agent.loss_history)), agent.loss_history)
plt.xlabel('Step')
plt.ylabel('Loss')
plt.title('Loss per Episode')
plt.show()
