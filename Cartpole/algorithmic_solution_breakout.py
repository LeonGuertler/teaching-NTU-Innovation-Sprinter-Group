import gym, time
import numpy as np # for observation manipulation

# create the environment
env = gym.make("Breakout-v0")

for game_nr in range(15):
    # initialize and reset game relevant variables
    score = action = done = 0
    observation = env.reset()

    # play until solved or failed
    while not done:

        # pre-process observation (rbg pixel data)
        observation = np.delete(observation, [56,57,58,59,60,61,62], axis=0)
        observation_c = np.sum(observation[:180],axis=-1)
        observation_d = np.sum(observation[180:],axis=-1)

        # find ball position based on colour
        a = np.zeros((list(np.shape(observation_c))))
        a[np.where(observation_c==np.sum([200,72,72]))] = 1
        ball_position = np.sum(a,axis=0)

        # find player position based on colour
        a = np.zeros((list(np.shape(observation_d))))
        a[np.where(observation_d==np.sum([200,72,72]))] = 1
        a[:, 150:] = 0
        player_position = np.sum(a,axis=0)

        # match x-coordinates of the ball and the player or use action 0 to start the game
        action = 2 if (np.argmax(player_position)+7) < (np.argmax(ball_position)) else 3
        action = 1 if np.sum(ball_position)==0 else action

        # execute the choosen action in the enviornment
        observation, reward, done, info = env.step(action)
        env.render()
        score += reward

        print(f"Game_nr: {game_nr}\t\tScore: {score}", end="\r")
    print(f"Game_nr: {game_nr}\t\tScore: {score}")
