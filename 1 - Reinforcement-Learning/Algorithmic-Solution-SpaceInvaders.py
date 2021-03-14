import matplotlib.pyplot as plt
import cv2
import numpy as np
import gym

env = gym.make("SpaceInvaders-v0")

def preprocess(obs):
    # grayscale
    obs = cv2.cvtColor(obs, cv2.COLOR_BGR2GRAY)
    return obs


done = score = 0
obs = env.reset()
frame = 0
while not done:
    obs = preprocess(obs)
    x_coo_p = np.where(obs[190,:]==98)

    if len(x_coo_p[0])!=0:
        x_val = x_coo_p[0][len(x_coo_p)//2]
        n_obs = obs[:, x_val-5:x_val+5]
        print(np.shape(n_obs))

    else:
        action = env.action_space.sample()

    action = env.action_space.sample()
    frame += 1
    if frame>100:
        plt.imshow(obs)
        plt.show()
    obs, _, done, _ = env.step(action)
