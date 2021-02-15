import numpy as np
import cv2

import gym, gym_snake
import time

import sample_ai

import matplotlib.pyplot as plt

"""
initialize relevant variables and CONSTANTS.
Team size of 2
"""
TOTAL_NR_GAMES = 1_000

NUM_TEAMS = 5
num_snakes = NUM_TEAMS * 2

x_size, y_size = 75, 75

env = gym.make("snake-plural-v0")
env.grid_size = [x_size, y_size]
env.n_snakes = num_snakes
env.n_foods = num_snakes

agent_list = []
agent_list.append(sample_ai.player_x(x_size=x_size, y_size=y_size))
agent_list.append(sample_ai.player_x(x_size=x_size, y_size=y_size))
agent_list.append(sample_ai.player_x(x_size=x_size, y_size=y_size))
agent_list.append(sample_ai.player_x(x_size=x_size, y_size=y_size))
agent_list.append(sample_ai.player_x(x_size=x_size, y_size=y_size))


def preprocess(obs):
    obs = cv2.cvtColor(obs, cv2.COLOR_BGR2GRAY)
    obs = cv2.resize(obs, (x_size, y_size), interpolation = cv2.INTER_AREA)
    #print(np.unique(obs))
    obs = (obs-np.min(obs)) / (np.max(obs)-np.min(obs))
    #plt.imshow(obs)
    #plt.show()
    return obs


k = 0
for game_nr in range(TOTAL_NR_GAMES):
    done = False
    obs = env.reset()
    obs = preprocess(obs)

    for agent in agent_list:
        agent.reset(train=True, test_obs=obs)

    while not done:
        action_list = np.zeros((num_snakes,))

        for x, agent in enumerate(agent_list):
            p_1 = x*2
            p_2 = x*2 + 1
            action_list[[p_1,p_2]] = agent.predict(obs=obs,
                                                   player_nr=[p_1,p_2],
                                                   train=True)

        obs, reward, done, info = env.step(action_list)
        obs = preprocess(obs)
        #print(obs)
        k+=1
        print(k)
        #print(reward)
        #env.render()

    for agent in agent_list:

        agent.reset(train=True, test_obs=obs)
        agent.train()