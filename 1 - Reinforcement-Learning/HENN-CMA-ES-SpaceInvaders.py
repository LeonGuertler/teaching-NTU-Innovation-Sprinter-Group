import numpy as np
import gym, time, cv2, os, random, cma
import tensorflow as tf

# for visualization
import matplotlib.pyplot as plt

env = gym.make("SpaceInvaders-ram-v0")
n = len(env.observation_space.sample())
num_labels = env.action_space.n

# Network Architecture
MAX_NUM_NEURONS = 24
NUM_CONNECTIONS_OUTPUT = np.max([int(MAX_NUM_NEURONS/(6*num_labels)), 1])

neuron_mask_shape = (MAX_NUM_NEURONS, (n+1)+MAX_NUM_NEURONS)
neuron_mask_length = np.product(list(neuron_mask_shape))

MAX_SCORE = 6_000


def get_init_solution():
    return np.concatenate([np.random.uniform(-.012,.012,neuron_mask_shape).reshape(-1),
                          np.random.uniform(-0.1, 0.1, size=neuron_mask_shape).reshape(-1)])

def initialize_output_layer_neurons():
    return np.random.choice(np.arange((n+1), (n+1)+MAX_NUM_NEURONS),
                            size=(num_labels,NUM_CONNECTIONS_OUTPUT),
                            replace=False)


def forward_pass(neuron_mask, output_neurons, neuron_structure, obs):
    neuron_structure[:(n+1)] = [1, *obs.copy()]
    neuron_structure[(n+1):] = np.tanh(np.dot(neuron_mask, neuron_structure))
    return np.argmax(np.sum(neuron_structure[output_neurons], axis=1))


def get_reward(neuron_mask):
    score = done = 0
    obs = env.reset()
    neuron_structure = np.zeros(((n+1)+MAX_NUM_NEURONS))
    while not done:
        action = forward_pass(neuron_mask=neuron_mask,
                              output_neurons=output_neurons,
                              neuron_structure=neuron_structure,
                              obs=obs)
        obs, reward, done, info = env.step(action)

        #done = True if info['ale.lives']!=3 else done
        score += reward
    return score

output_neurons = initialize_output_layer_neurons()
x0 = get_init_solution()


episode_count = 0
score_list = []
fitness_list = []
size_list = []

def f(x):
    global episode_count
    x = x.copy()
    rndm_mask = np.random.uniform(size=neuron_mask_shape)
    x = x[neuron_mask_length:].reshape(neuron_mask_shape)*\
        (np.abs(x[:neuron_mask_length].reshape(neuron_mask_shape))>rndm_mask)

    Reward = get_reward(x)
    num_neurons = np.sum(x!=0)

    fitness_value = -(Reward/MAX_SCORE) * 100 + (num_neurons/(MAX_NUM_NEURONS*(n+MAX_NUM_NEURONS)))
    if not (episode_count%25):
        print(f"{episode_count}:\tReward: {Reward:.2f}\tnum_neurons: {num_neurons}\tfitness_value: {fitness_value:.2f}")

    score_list.append(Reward)
    fitness_list.append(fitness_value)
    size_list.append(num_neurons)
    episode_count += 1
    return fitness_value


options = {}
options['maxfevals']	= 1e6 # 10000

options['tolx']		= 0
options['tolfun']	= 0
options['tolfunhist']	= 0

noise_handler = cma.NoiseHandler(len(x0))

#with tf.device('/GPU:0'):
cma.fmin(f, x0, .2, options, noise_handler=noise_handler)


np.save("score_list.npy", np.asarray(score_list))
np.save("fitness_list.npy", np.asarray(fitness_list))
np.save("size_list.npy", np.asarray(size_list))


fig, ax = plt.subplots(3)
ax[0].title.set_text("Score")
fig.suptitle("attempt_1")
ax[0].plot(np.convolve(score_list, np.ones(100), 'valid') / 100)#score_list)

ax[1].title.set_text("fitness")
ax[1].plot(np.convolve(fitness_list, np.ones(100), 'valid') / 100)#fitness_list)

ax[2].title.set_text("size")
ax[2].plot(np.convolve(size_list, np.ones(100), 'valid') / 100)#size_list)

plt.show()
