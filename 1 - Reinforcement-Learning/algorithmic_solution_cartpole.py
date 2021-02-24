import gym, time

# create the environemt
env = gym.make("CartPole-v1")

for game_nr in range(15):
    # initialize and reset game relevant variables
    score = action = frame = done = 0
    observation = env.reset()

    # play until solved or failed
    while not done:

        # the following if statements determine the action and are based on intuition
        if observation[3]>.09:
            action = 1
        if observation[3]<-.09:
            action = 0
        if not (frame%5):
            if observation[1]>.45:
                action=1
            elif observation[1]<-.45:
                action=0

        # execute the choosen action in the enviornment
        observation, reward, done, info = env.step(action)
        env.render()
        score += reward
        frame += 1

        print(f"Game_nr: {game_nr}\t\tScore: {score}", end="\r")
    print("")
