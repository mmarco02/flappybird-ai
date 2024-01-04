# Flappy Bird Reinforcement Learning Environment
This repository contains a custom Gym environment for training reinforcement learning agents on the Flappy Bird game. The environment is implemented using OpenAI Gym's gym.Env base class and integrates the Flappy Bird game through the flappybird module.

## FlappyBirdEnvironment
### Installation
Before using this environment, ensure you have the required dependencies installed. You can install them using the following:

```bash
Copy code
pip install gym
pip install numpy
```

Usage:
To use this environment in your reinforcement learning project, you can follow these steps:

#### Import the environment:
```python
Copy code
from flappybird_environment import FlappyBirdEnvironment
```
#### Create an instance of the environment:
```python
Copy code
env = FlappyBirdEnvironment()
```

Interact with the environment using the standard Gym interface (reset, step, etc.):
```python
Copy code
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
Copy code
reward = self._flappy_bird_game.score + (self._flappy_bird_game.distance / 100)
```
Additionally, if the game ends, a penalty of -10 is applied.

### Example
```python
Copy code
env = FlappyBirdEnvironment()

# Reset the environment to start a new episode
observation, _ = env.reset()

for _ in range(1000):
    # Take a random action
    action = env.random_action()

    # Interact with the environment
    observation, reward, done, _ = env.step(action)

    if done:
        print("Episode ended!")
        break
```
