# Leon Gurtler
import gym, time
import numpy as np

# for visualization
from tabulate import tabulate
import matplotlib.pyplot as plt
plt.style.use("seaborn")

env = gym.make("CartPole-v0")
n = len(env.observation_space.sample())
num_labels = env.action_space.n


"""
# Q-Table Parameters etc.

This implementation does purposefully not use epsilon greedy or other exploration
techniques to showcase the need for such! (aka I was lazy)


The environment itself has restrictions on the observation_space
    0 (Cart Position):  Min (-2.4)  Max (2.4)
    1 (Cart Velocity):  Min (-inf)  Max (inf)
    2 (Pole Angle):     Min (-41.8) Max (41.8)
    3 (Pole Velocity):  Min (-inf)  Max (inf)

The key idea is that we, in an attempt to reduce the size of the Q-Table, only check
whether a specific observation is positive or negative. -> 32 elements in the table.
This will not yield the highest score, but gives and intuitive understanding of
how Q-Learning works and how it can be applied.
"""

LR = .1
q_table = np.random.uniform(0,1,size=(2,2,2,2,2))
print("The Q_table has: ", np.product(list(np.shape(q_table))), "elements.")


def preprocess(obs):
    """
    The preprocessing is necessary to reduce the total observation space into
    something that our Q-Table is able to handle (hence, not too many different
    possibilities).
    """
    th_value = 0.04 # pretty arbitrary
    obs = np.asarray(obs)
    obs[np.where(obs>th_value)] = 1
    obs[np.where(obs<th_value)] = 0
    return obs.astype(int)



TOTAL_GAMES = 10_000 # arbitrary
score_list = []

for game_nr in range(TOTAL_GAMES):
    done = score = 0    # in python 0 == False
    obs = env.reset()

    # initialize game_memory to keep track of the obs, action pairs for training
    game_memory_obs = []
    game_memory_action = []


    while not done:
        obs = preprocess(obs)
        action = np.argmax(q_table[obs[0], obs[1], obs[2], obs[3]])

        # append the current memory for local training
        game_memory_obs.append(obs)
        game_memory_action.append(action)

        obs, reward, done, info = env.step(action)
        score += reward


    score_list.append(score) # for a pretty plot at the end
    if not (game_nr%50): # print every n games
        print(f"{game_nr} / {TOTAL_GAMES}"+\
              f"\t Mean: {np.mean(score_list):.2f}"+\
              f"\t Mean (past 200): {np.mean(score_list[-200:]):.2f}"+\
              f"\t Max: {np.max(score_list)}", end="\r")

    score /= 200
    for obs,action in zip(game_memory_obs, game_memory_action):
        q_table[obs[0], obs[1], obs[2], obs[3], action] = (1-LR)*q_table[obs[0], obs[1], obs[2], obs[3], action] + \
                                                            LR * score


# a lot of effort for a small print (prints the actual Q-Table (simplified))
tabulated_list = []
counter = 0
for obs_1 in range(len(q_table[0])):
    for obs_2 in range(len(q_table[0][0])):
        for obs_3 in range(len(q_table[0][0][0])):
            for obs_4 in range(len(q_table[0][0][0][0])):
                tabulated_list.append([f"Observation_{counter}", *q_table[obs_1][obs_2][obs_3][obs_4]])
                counter += 1
print(tabulate(tabulated_list, headers=["Q-Table", "Action_1", "Action_2"]))


# pretty plot with raw score and moving averages across 100 and 500 games.
plt.title("Q-Learning")
plt.plot(score_list, label="Score", alpha=.25)
plt.plot(np.arange(100-1, len(score_list)), np.convolve(score_list,np.ones(100), 'valid') / 100, label="ma_100")
plt.plot(np.arange(500-1, len(score_list)), np.convolve(score_list,np.ones(500), 'valid') / 500, label="ma_500")
plt.legend()
plt.show()

# play a few games with the final Q-Table
for _ in range(25):
    done = score = 0
    obs = env.reset()
    while not done:
        obs = preprocess(obs)
        action = np.argmax(q_table[obs[0], obs[1], obs[2], obs[3]])
        obs, reward, done, info = env.step(action)
        env.render()    # actually render the games
        score += reward
        print(score, end="\r")
    print("") # necessary since we use end="\r" above but don't want to overwrite
