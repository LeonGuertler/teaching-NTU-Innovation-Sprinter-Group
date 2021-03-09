# Leon Gurtler
import gym, time
import numpy as np

env = gym.make("SpaceInvaders-v0")
n = np.shape(env.observation_space.sample())
num_labels = env.action_space.n

print(f"Observation shape:\t{n}")
print(f"Action space:\t\t{num_labels}")


# play 3 random games
for _ in range(3):
    done = 0
    env.reset()
    while not done:
        action = env.action_space.sample()
        _, _, done, _ = env.step(action)
        env.render()
        time.sleep(.01)

input("Test the individual actions....(press Enter)")
# test the individual actions
for action in range(num_labels):
    done = 0
    env.reset()
    print(f"Testing Action: {action}")
    while not done:
        _,_,done,_ = env.step(action)
        env.render()
        time.sleep(.01)

"""
Action-space:
    0 - do nothing
    1 - shoot
    2 - move right
    3 - move left
    4 - move right & shoot
    5 - move left & shoot
"""

input("Ready to see the objectively beautiful pre-processing?")
import matplotlib.pyplot as plt
import cv2

def preprocess(obs):
    plt.imshow(obs)
    plt.title("Raw")
    plt.show()

    # grayscale
    obs = cv2.cvtColor(obs, cv2.COLOR_BGR2GRAY)
    plt.imshow(obs, cmap='gray', vmin=0, vmax=255)
    plt.title("Grayscaled")
    plt.show()


    # resize
    obs = cv2.resize(obs, (84,84))
    print(f"Final shape: {np.shape(obs)}")
    plt.imshow(obs, cmap='gray', vmin=0, vmax=255)
    plt.title("Grayscaled & resized")
    plt.show()



env.reset()
obs, reward, done, info = env.step(env.action_space.sample())
obs = preprocess(obs)



input("Same thing for the ram environment")


env = gym.make("SpaceInvaders-ram-v0")
n = np.shape(env.observation_space.sample())
num_labels = env.action_space.n

print(env.observation_space.sample())
print(f"Observation shape:\t{n}")
print(f"Action space:\t\t{num_labels}")
env.close()
