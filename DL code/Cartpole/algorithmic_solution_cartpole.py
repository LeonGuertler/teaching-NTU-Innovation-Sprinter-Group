import gym, time
env = gym.make("CartPole-v1")

for _ in range(15):
    score = action = frame = done = 0
    observation = env.reset()
    while not done:
        if observation[3]>.09:
            action = 1
        if observation[3]<-.09:
            action = 0
        if not (frame%5):
            if observation[1]>.45:
                action=1
            elif observation[1]<-.45:
                action=0
        observation, reward, done, info = env.step(action)
        env.render()
        score += reward
        frame += 1
        print(score, "\t", observation[1])
