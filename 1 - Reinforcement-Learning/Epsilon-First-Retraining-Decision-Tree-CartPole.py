# Leon Gurtler
import gym, time
import numpy as np

from sklearn.tree import DecisionTreeClassifier

env = gym.make("CartPole-v0")
n = len(env.observation_space.sample())
num_labels = env.action_space.n


# To save the training data
X_train = []
y_train = []

score_threshold = 30 # everything above this threshold is appended for training
TOTAL_GAMES = 10_000 # the total nr of games played
TRAIN_EVERY = 100
nr_games_appended = 0 # keep track of nr of games appended to training

clf = DecisionTreeClassifier()
score_list = []
trained = False

for game_nr in range(TOTAL_GAMES):
    done = score = 0 # in python 0 == False
    obs = env.reset()

    # initialize local game_based memory
    game_obs = []
    game_action = []

    while not done:
        if not trained:
            action = env.action_space.sample() # choose a random action
        else:
            action = clf.predict(np.asarray([obs])).astype(int)[0]

        # append the observation, action pair to the local memory
        game_obs.append(obs)
        game_action.append(action)

        # execute the action
        obs, reward, done, info = env.step(action)
        score += reward

    score_list.append(score)
    if score >= score_threshold: # if score above threshold, append
        nr_games_appended += 1
        X_train+=game_obs
        y_train+=game_action

    if not (game_nr%10) or trained: # print every n games
        print(f"{game_nr} / {TOTAL_GAMES}\tscore: {score}\tthreshold: {score_threshold:.2f}\tBatch avg. {np.mean(score_list):.2f}\tnr games appended: {nr_games_appended}", end="\r")

    if not ((game_nr+1)%TRAIN_EVERY):
        print(f"\nTraining:\tmax score: {np.max(score_list)}\tmean score: {np.mean(score_list):.2f}")
        # train the RandomForest
        clf.fit(
            np.asarray(X_train),
            np.asarray(y_train)
        )
        score_threshold = np.max([np.percentile(score_list, 75), score_threshold])
        nr_games_appended = 0
        trained = True
        X_train = []
        y_train = []
        score_list = []



# use the trained network to play
for _ in range(25):
    done = score = 0
    obs = env.reset()
    while not done:
        action = clf.predict(np.asarray([obs])).astype(int)[0]
        obs, reward, done, info = env.step(action)
        env.render()    # actually render the games
        score += reward
        print(score, end="\r")
    print("") # necessary since we use end="\r" above but don't want to overwrite
