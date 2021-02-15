import numpy as np


import gym_snake
import gym, time

import matplotlib.pyplot as plt


NUM_SNAKES = 2*5

env = gym.make("snake-plural-v0")
x_size,y_size = 75, 75
env.grid_size = [x_size,y_size]
env.n_snakes = NUM_SNAKES
env.n_foods = NUM_SNAKES

action_list = np.arange(env.action_space.n)
#input(action_list)
k = 0
for _ in range(10):
    done = False
    env.reset()

    while not done:
        action = env.action_space.sample()

        #print(action)

        observation,_,done,_ = env.step(
            np.random.choice(action_list,size=(NUM_SNAKES,))
        )
        #env.render()
        #print(observation)
        #print(np.shape(observation))
        #print(np.unique(observation))

        k += 1
        print(k)



        # preprocessing
        #obs = np.sum(observation, axis=-1)
        #obs = (obs-np.min(obs)) / (np.max(obs)-np.min(obs))
        #plt.imshow(obs)
        #plt.show()

        #obs_2 = obs[np.arange(0,x_size*10,10)][:, np.arange(0, y_size*10, 10)]

        #print(obs_2)
        #plt.imshow(obs_2)
        #plt.show()




        #input("ok")
        #input(" ")
        #time.sleep(0.09)


"""
grid_size = [25,25]
unit_size = 10
unit_gap = 1
snake_size = 5
n_snakes = 3
n_foods = 2
"""
