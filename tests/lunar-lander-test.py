import gym
from spatialNN import Model

env = gym.make("LunarLander-v2")


observation = env.reset()
agent.setInputShape(observation)