# Leon Gurtler
import gym, time
import numpy as np

from sklearn.ensemble import RandomForestClassifier

env = gym.make("CartPole-v0")
n = len(env.observation_space.sample())
num_labels = env.action_space.n


# To save the training data
X_train = []
y_train = []

SCORE_THRESHOLD = 75 # everything above this threshold is appended for training
INITIAL_GAMES = 100_000 # the nr of random games played initially
nr_games_appended = 0 # keep track of nr of games appended to training

clf = RandomForestClassifier()

for game_nr in range(INITIAL_GAMES):
    done = score = 0 # in python 0 == False
    observation = env.reset()

    # initialize local game_based memory
    game_obs = []
    game_action = []

    while not done:
        action = env.action_space.sample() # choose a random action

        # append the observation, action pair to the local memory
        game_obs.append(observation)
        game_action.append(action)

        # execute the action
        observation, reward, done, info = env.step(action)
        score += reward


    if score >= SCORE_THRESHOLD: # if score above threshold, append
        nr_games_appended += 1
        X_train+=game_obs
        y_train+=game_action

    if not (game_nr%50): # print every n games
        print(f"{game_nr} / {INITIAL_GAMES}\t\tnr games appended: {nr_games_appended}", end="\r")


# train the RandomForest
clf.fit(
    np.asarray(X_train),
    np.asarray(y_train)
)


# use the trained network to play
for _ in range(25):
    done = score = 0
    obs = env.reset()
    while not done:
        action = clf.predict(np.asarray([obs])).astype(int)[0]#>.5
        obs, reward, done, info = env.step(action)
        env.render()    # actually render the games
        score += reward
        print(score, end="\r")
    print("") # necessary since we use end="\r" above but don't want to overwrite
