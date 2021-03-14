import numpy
import keras
import gym
import cma


environment = gym.make("SpaceInvaders-ram-v0")
environment.frameskip = 4

model = keras.models.Sequential([
  keras.layers.Dense(64,  activation="tanh", input_shape=environment.observation_space.shape),
  keras.layers.Dense(64,  activation="tanh"),
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
    action = get_action(observation/255.0)
    observation, reward, done, _info = environment.step(action)
    Reward += reward
  print(Reward)
  return Reward

def f(x):
  set_weights(x)
  Reward = get_reward()
  return -Reward

x0 = get_solution(model.get_weights())


options = {}

options['maxfevals']  = 1e4

options['tolx']    = 0
options['tolfun']  = 0
options['tolfunhist']  = 0

noise_handler = cma.NoiseHandler(len(x0))
cma.fmin(f, x0, 1.0, options, noise_handler=noise_handler)
