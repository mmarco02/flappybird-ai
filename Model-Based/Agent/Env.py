import gym

gym.register(id='FlappyBird-m', entry_point='flappy_bird_gym.envs.flappy_bird_env:FlappyBirdEnvironment')

env = gym.make('FlappyBird-m')