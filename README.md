# Flappy Bird Reinforcement Learning Environment
This repository contains a custom Gym environment for training reinforcement learning agents on the Flappy Bird game. The environment is implemented using OpenAI Gym's gym.Env base class and integrates the Flappy Bird game.

## FlappyBirdEnvironment
### Installation
Before using this environment, ensure you have the required dependencies installed. You can install them using the following:

```bash
pip install gym
pip install numpy
pip install tensorflow
```

Usage:
To use this environment in your reinforcement learning project, you can follow these steps:

#### Import the environment:
```python
from flappy_bird_gym.envs.flappy_bird_env import FlappyBirdEnvironment
import gym
```
#### Create an instance of the environment:
```python
# Register the environment
gym.register(id='FlappyBird-m', entry_point='flappy_bird_gym.envs.flappy_bird_env:FlappyBirdEnvironment')

# Create an instance using gym.make
flappy_env = gym.make('FlappyBird-m')
```

Interact with the environment using the standard Gym interface (reset, step, etc.):
```python
observation, reward, done, info = env.step(action)
```

#### Environment Details
Observation Space
The observation space is a continuous vector of five elements, representing the following features:

- Bird's Y-coordinate
- Bird's velocity
- Distance to the next pipe
- Vertical distance to the next top pipe
- Vertical distance to the next bottom pipe

#### Action Space
The action space is discrete with two possible actions:

0: Do nothing
1: Make the bird jump


### Rewards
The agent receives a reward based on the game's score and the distance traveled by the bird. The reward calculation is as follows:

```python
reward = self._flappy_bird_game.score + (self._flappy_bird_game.distance / 100)
```
Additionally, if the game ends, a penalty of -10 is applied.

### Example
```python
import gym
from flappy_bird_gym.envs.flappy_bird_env import FlappyBirdEnvironment
from flappybird import FlappyBird

# Register the environment
gym.register(id='FlappyBird-m', entry_point='flappy_bird_gym.envs.flappy_bird_env:FlappyBirdEnvironment')

# Create an instance using gym.make
flappy_env = gym.make('FlappyBird-m')

# Use the environment as usual
initial_observation, _ = flappy_env.reset()
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
```
