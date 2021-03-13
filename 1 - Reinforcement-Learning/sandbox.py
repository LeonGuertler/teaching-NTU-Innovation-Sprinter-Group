import argparse
import numpy
import keras
import gym
import cma

parser = argparse.ArgumentParser()
parser.add_argument("environment")
args = parser.parse_args()

environment = gym.make("CartPole-v0")

model = keras.models.Sequential([
  keras.layers.Dense(10,  activation="tanh", input_shape=environment.observation_space.shape),
  keras.layers.Dense(5,  activation="tanh"),
  keras.layers.Dense(environment.action_space.n)])

shapes = [weight.shape for weight in model.get_weights()]

def get_solution(weights):
  return numpy.concatenate([weight.reshape(-1) for weight in weights])

def set_weights(solution):
  model.set_weights([solution[1:1+numpy.prod(shape)].reshape(shape) for shape in shapes])

def get_action(observation):
  return numpy.argmax(model.predict_on_batch(observation))

shape = (1,) + environment.observation_space.shape

def get_reward():
  observation = environment.reset()
  Reward = 0
  done = False
  while not done:
    observation = observation.reshape(shape)
    action = get_action(observation)
    observation, reward, done, _info = environment.step(action)
    Reward += reward
  return Reward

def f(x):
  set_weights(x)
  Reward = get_reward()
  return -Reward

x0 = get_solution(model.get_weights())
