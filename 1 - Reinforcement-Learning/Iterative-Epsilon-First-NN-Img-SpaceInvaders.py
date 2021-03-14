# Leon Gurtler
import gym, time, cv2
import numpy as np

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, Flatten


env = gym.make("SpaceInvaders-v0")
env.frameskip = 4
n = np.shape(env.observation_space.sample())
num_labels = env.action_space.n

# To save the training data
X_train = []
y_train = []

# variables and CONSTANTS
epsilon = .5
EPSILON_DECAY = .95
TOTAL_GAMES = 100_000

score_list = []
score_threshold = 150



# probably a good idea to initialize your model somewhere here
model = Sequential()
model.add(Conv2D(32, 8, 4, activation="tanh", input_shape=(84,84,1)))
model.add(MaxPooling2D())

model.add(Conv2D(64, 4, 2, activation="tanh"))
model.add(MaxPooling2D())

model.add(Conv2D(64, 3, 1, activation="tanh"))
model.add(MaxPooling2D())

model.add(BatchNormalization())
model.add(Flatten())

model.add(Dense(512, activation="relu"))

model.add(Dense(num_labels))

model.compile(loss='mean_squared_error',
                   optimizer='rmsprop',
                   metrics=['accuracy'])

def preprocess(obs):
    """ Basic pre-process to grayscale and re-shape for (84,84,1) or (84,84) """
    obs = cv2.cvtColor(obs, cv2.COLOR_BGR2GRAY) # grayscale
    obs = cv2.resize(obs, (84,84))/255.0 # resize
    return obs[...,np.newaxis] # necessary for Conv2D Layers

trained = False
# training loop
for game_nr in range(TOTAL_GAMES):
    done = score = 0
    obs = env.reset()

    # initialize local game_based memory
    game_obs = []
    game_action = []

    while not done:
        obs = preprocess(obs)

        if np.random.uniform() < epsilon or not trained: # basic epsilon-decreasing/-greedy strategy
            action = env.action_space.sample()
        else:

            action = np.argmax(model.predict(np.asarray([obs]))[0])

        game_obs.append(obs)
        game_action.append(action)
        obs, reward, done, info = env.step(action)
        score+= reward

    score_list.append(score)

    if score >= score_threshold:
        for obs, a in zip(game_obs, game_action):
            label = np.zeros((num_labels,))
            label[a] = 1#score
            y_train.append(label)
            X_train.append(obs)


    #if not (game_nr%50):
    print(f"{game_nr} / {TOTAL_GAMES}"+\
          f"\tMost recent score: {score}"+\
          f"\tInter-training score-avg: {np.mean(score_list)}", end="\r")



    if not (game_nr+1)%200:
        trained = True
        print(f"{game_nr} / {TOTAL_GAMES}"+\
              f"\tTraining_batch avg. score: {np.mean(score_list)}"+\
              f"\tepsilon: {epsilon}"+\
              f"\tScore Threshold: {score_threshold}")
        model.fit(
            np.asarray(X_train),
            np.asarray(y_train),
            epochs=5
        )
        X_train = []
        y_train = []
        score_threshold = np.percentile(np.asarray(score_list), 60)
        score_list = []
        epsilon *= EPSILON_DECAY
        #model.save("myAwesomeModelLah.model")
    #"""
