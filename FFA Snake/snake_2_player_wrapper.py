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
TOTAL_NR_GAMES = 100_000

NUM_TEAMS = 1
num_snakes = NUM_TEAMS * 2

x_size, y_size = 75, 75

env = gym.make("snake-plural-v0")
env.grid_size = [x_size, y_size]
env.n_snakes = num_snakes
env.n_foods = num_snakes

agent_list = []
agent_list.append(sample_ai.player_x(x_size=x_size, y_size=y_size))
"""agent_list.append(sample_ai.player_x(x_size=x_size, y_size=y_size))
agent_list.append(sample_ai.player_x(x_size=x_size, y_size=y_size))
agent_list.append(sample_ai.player_x(x_size=x_size, y_size=y_size))
agent_list.append(sample_ai.player_x(x_size=x_size, y_size=y_size))"""


def preprocess(obs):
    obs = cv2.cvtColor(obs, cv2.COLOR_BGR2GRAY)
    obs = cv2.resize(obs, (x_size, y_size), interpolation = cv2.INTER_AREA)
    obs = (obs-np.min(obs)) / (np.max(obs)-np.min(obs))
    return obs
k = 0
trained = False
for game_nr in range(TOTAL_NR_GAMES):
    done = False
    obs = env.reset()
    obs = preprocess(obs)
    print(k)
    k = 0


    for agent in agent_list:
        agent.reset(train=True, test_obs=obs)

    while not done:
        action_list = np.zeros((num_snakes,))

        for x, agent in enumerate(range(len(agent_list))):
            p_1 = x*2
            p_2 = x*2 + 1
            action_list[[p_1,p_2]] = agent_list[x].predict(obs=obs,
                                                   player_nr=[p_1,p_2],
                                                   train=True)

        obs, reward, done, info = env.step(action_list)
        obs = preprocess(obs)
        #print(obs)
        k+=1
        if not (k%20):
            print(f"{k}", end='\r')
        #print(reward)
        #env.render()
        if k > 2500:
            env.render()

    for agent in agent_list:

        agent.reset(train=True, test_obs=obs)
        agent.train()
