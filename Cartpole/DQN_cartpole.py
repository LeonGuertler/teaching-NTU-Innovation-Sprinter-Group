import gym, time
import numpy as np

import keras
from keras.layers import Dense, Dropout
from keras.models import Sequential


env = gym.make("CartPole-v1")
n = len(env.observation_space.sample())
num_labels = env.action_space.n


X_train = []
y_train = []

epsilon = .5
EPSILON_DECAY = .9

TRAIN_SIZE = 20_000
TOTAL_GAMES = 200_000

model = Sequential()

model.add(Dense(128, activation="relu", input_shape=(n,)))
model.add(Dropout(.5))

model.add(Dense(256, activation="relu"))
model.add(Dropout(.5))

model.add(Dense(512, activation="relu"))
model.add(Dropout(.5))

model.add(Dense(256, activation="relu"))
model.add(Dropout(.5))

model.add(Dense(128, activation="relu"))
model.add(Dropout(.5))

model.add(Dense(2, activation="linear"))

model.compile(loss="mse", optimizer="adam", metrics=["accuracy"])




for game_nr in range(TOTAL_GAMES):
    done = False
    observation = env.reset()
    score = 0

    game_obs = []
    game_action = []

    while not done:
        if game_nr<500 or np.random.uniform()<=epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(model.predict(np.asarray([observation]))[0])

        game_obs.append(observation)
        game_action.append(action)
        try:
            observation, reward, done, info = env.step(action)
        except:
            print(model.predict(np.asarray([observation]))[0])
            print(np.argmax(model.predict(np.asarray([observation]))[0]))
            print(np.argmax(model.predict(np.asarray([observation]))[0]).dtype)
            observation, reward, done, info = env.step(action.item())
        score += reward

    if not (game_nr%25):
        print(f"{game_nr} / {TOTAL_GAMES}\t\t{score}\t\t{len(y_train)}", end='\r')


    for obs,a in zip(game_obs, game_action):
        label = np.zeros((2,))
        label[a] = score / 500.0
        X_train.append(obs)
        y_train.append(label)




    if len(y_train)>=TRAIN_SIZE:
        model.fit(np.asarray(X_train), np.asarray(y_train), epochs=3)
        epsilon *= EPSILON_DECAY
        X_train = []
        y_train = []
        print("")
