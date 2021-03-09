# Leon Gurtler
import gym, time
import numpy as np

import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.utils import class_weight

env = gym.make("CartPole-v0")
n = len(env.observation_space.sample())
num_labels = env.action_space.n


# To save the training data
X_train = []
y_train = []

TOTAL_GAMES = 100_000 # the total nr of games played
TRAIN_EVERY = 1_000
epsilon = .4
EPSILON_DECAY = .85


# Neural Network
model = Sequential()
model.add(Dense(24, activation="relu", input_dim=(4)))
model.add(Dense(48, activation="relu"))
model.add(Dense(96, activation="relu"))
model.add(Dense(48, activation="relu"))
model.add(Dense(24, activation="relu"))
model.add(Dense(2, activation="relu"))
model.compile(loss="mse", optimizer="adam")

score_list = []
trained = False
for game_nr in range(TOTAL_GAMES):
    done = score = 0 # in python 0 == False
    obs = env.reset()

    # initialize local game_based memory
    game_obs = []
    game_action = []

    while not done:
        if not trained or epsilon>np.random.uniform():
            action = env.action_space.sample() # choose a random action
        else:
            action = np.argmax(model.predict(np.asarray([obs]))[0])

        # append the observation, action pair to the local memory
        game_obs.append(obs)
        game_action.append(action)

        # execute the action
        obs, reward, done, info = env.step(action)
        score += reward

    score_list.append(score)
    game_action=np.asarray(game_action)
    min_action_qty = np.min([np.sum(game_action==0), np.sum(game_action==1)])
    action_count = np.ones((2))*min_action_qty
    for a, obs in zip(game_action, game_obs):
        if action_count[a]>0:
            label = np.zeros((2,))
            label[a]=score
            y_train.append(label)
            X_train.append(obs)
            action_count[a]-=1
    #X_train+=game_obs

    if not (game_nr%50) or trained: # print every n games
        print(f"{game_nr} / {TOTAL_GAMES}\tscore: {score}\tBatch avg. {np.mean(score_list):.2f}", end="\r")

    if not ((game_nr+1)%TRAIN_EVERY):
        epsilon *= EPSILON_DECAY
        print(f"\nTraining:\tmax score: {np.max(score_list)}\tmean score: {np.mean(score_list):.2f}\tepsilon: {epsilon}")
        # train the Neural Network
        model.fit(
            np.asarray(X_train),
            np.asarray(y_train),
            epochs=10
        )
        trained = True
        X_train = []
        y_train = []
        score_list = []



# use the trained network to play
for _ in range(25):
    done = score = 0
    obs = env.reset()
    while not done:
        action = np.argmax(model.predict(np.asarray([obs]))[0])
        obs, reward, done, info = env.step(action)
        env.render()    # actually render the games
        score += reward
        print(score, end="\r")
    print("") # necessary since we use end="\r" above but don't want to overwrite
