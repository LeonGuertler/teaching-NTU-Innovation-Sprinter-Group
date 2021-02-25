# Leon Gurtler
import gym, time, random
import numpy as np

# for visualization
from tabulate import tabulate
import matplotlib.pyplot as plt
plt.style.use("seaborn")

env = gym.make("CartPole-v0")
n = len(env.observation_space.sample())
num_labels = env.action_space.n


# GA CONSTATNS and variables
GENERATION_SIZE = 128
EPOCHS = 35
KEEP_BEST_N = 10
ASSESS_N_GAMES = 3

MUTATION_RATE = .5
MUTATION_PROB = .7


# initiate the q-table
q_table_struct = np.random.uniform(0,1,size=(GENERATION_SIZE, 2,2,2,2,2))


def preprocess(obs):
    """
    The preprocessing is necessary to reduce the total observation space into
    something that our Q-Table is able to handle (hence, not too many different
    possibilities).
    """
    th_value = 0.03 # pretty arbitrary
    obs = np.asarray(obs)
    obs[np.where(obs>th_value)] = 1
    obs[np.where(obs<th_value)] = 0
    return obs.astype(int)


def evaluate_fitness(q_table):
    """
    Evaluate the fitness for a specific q-table.
    """
    global ASSESS_N_GAMES
    total_score = 0
    for _ in range(ASSESS_N_GAMES):
        done = 0
        obs = env.reset()
        while not done:
            obs = preprocess(obs)
            action = np.argmax(q_table[obs[0], obs[1], obs[2], obs[3]])
            obs, reward, done, info = env.step(action)
            total_score += reward
    return total_score/ASSESS_N_GAMES

def repopulate(fittest_n_genes):
    """
    After the performance of all genes has been assessed, we take the best n genes
    and use them to re-populate our generation.
    """
    global MUTATION_RATE, MUTATION_PROB, GENERATION_SIZE

    q_table_shape = np.shape(fittest_n_genes[0])
    new_generation_shape = (GENERATION_SIZE, *list(q_table_shape))
    new_generation = np.zeros((new_generation_shape))

    # keep the fittest_n_genes in the next generation (for robustness)
    new_generation[:len(fittest_n_genes)] = fittest_n_genes.copy()

    # For all other genes required to fill-up the new generation, pick two random
    # ones from the fittest_n_genes, and combine them by choosing about 50% of all
    # values from each. Broaded the exploration space by introductin the mutation-mask
    # (as common for darwian evolution). It basically introduces random small changes.
    for x in range(len(fittest_n_genes), GENERATION_SIZE):
        parent_mask = np.random.uniform(size=q_table_shape)>.5
        mutation_mask = np.random.uniform(1-MUTATION_RATE, 1+MUTATION_RATE, size=(q_table_shape)) * \
                        (np.random.uniform(size=(q_table_shape))<MUTATION_PROB)

        new_generation[x] = (random.choice(fittest_n_genes) * parent_mask + \
                            random.choice(fittest_n_genes) * (~parent_mask)) * \
                            mutation_mask
    return new_generation




score_matrix = np.zeros((EPOCHS, GENERATION_SIZE))
for epoch in range(EPOCHS):
    assessment_vector = np.zeros((len(q_table_struct)))
    for x in range(len(q_table_struct)):
        assessment_vector[x] = evaluate_fitness(q_table=q_table_struct[x])

    # this is used to keep track of results for post-training analysis
    score_matrix[epoch] = assessment_vector[:GENERATION_SIZE].copy()
    print(f"{epoch} / {EPOCHS}\t"+\
          f"\tepoch_max: {np.max(assessment_vector):.0f}\t"+\
          f"\tepoch_mean: {np.mean(assessment_vector):.0f}\t"+\
          f"\tgames_played: {epoch*GENERATION_SIZE*ASSESS_N_GAMES}")#, end="\r")


    # pick the cut-off point for score and pass all genes with a greater or equal
    # score into the re-population function.
    best_nth_value = np.sort(assessment_vector)[-KEEP_BEST_N]
    q_table_struct = repopulate(
        fittest_n_genes=q_table_struct[np.where(assessment_vector>=best_nth_value)[:KEEP_BEST_N]]
        )


# create a pretty plot eh
plt.title("Q-Learning GA")
plt.plot(np.max(score_matrix,axis=1), label="max")
plt.plot(np.mean(score_matrix,axis=1), label="mean")
plt.plot(np.median(score_matrix,axis=1), label="median")
plt.plot(np.min(score_matrix,axis=1), label="min")
plt.legend()
plt.show()


# choose the best gene of the last epoch to render a few games
q_table = q_table_struct[np.argmax(assessment_vector)]
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
