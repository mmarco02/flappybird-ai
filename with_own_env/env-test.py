import gym
from flappy_bird_gym.envs.flappy_bird_env import FlappyBirdEnvironment
from flappybird import FlappyBird

# Register the environment
gym.register(id='FlappyBird-m', entry_point='flappy_bird_gym.envs.flappy_bird_env:FlappyBirdEnvironment')

# Create an instance using gym.make
flappy_env = gym.make('FlappyBird-m', fbgame=FlappyBird())

# Use the environment as usual
initial_observation = flappy_env.reset()
print("Initial Observation:", initial_observation)

# Perform some steps in the environment
for _ in range(10):
    action = flappy_env.action_space.sample()
    observation, reward, done, info = flappy_env.step(action)
    print("Observation:", observation)
    print("Reward:", reward)
    print("Done:", done)
    print("Info:", info)

    if done:
        print("Episode ended. Resetting environment.")
        flappy_env.reset()

# Optionally, close the environment when you're done
flappy_env.close()
