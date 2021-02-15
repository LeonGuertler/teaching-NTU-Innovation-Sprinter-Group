import numpy as np
import cv2, h5py
import matplotlib.pyplot as plt



class player_x():
    def __init__(self, x_size, y_size):
        self.x_size = x_size
        self.y_size = y_size

        self.player_1_colour = 77
        self.player_2_colour = 99
        self.enemy_colour = 22

        self.frames = 0
        self.player_1_alive = True
        self.player_2_alive = True


        self.X_train = []
        self.y_train = []


        self.obs_memory = []
        self.action_memory = []

        self.weight_name = "TEAM_NAME.h5p"


    def reset(self, train, test_obs):
        if self.frames>0 and train:
            # append to training set (or not, kinda up to you man, don't @ us)
            for obs,a in zip(self.obs_memory, self.action_memory):
                self.X_train.append(obs)
                label = np.zeros((8))
                label[a] = self.frames
                self.y_train.append(label)

        self.frames = 0
        self.player_1_alive = True
        self.player_2_alive = True


        # body = 0
        # bg = 1
        # food = 0.516
        #print(np.unique(test_obs))
        self.player_colour_list = [e for e in np.unique(test_obs) if not (e in [0, 1, 0.5161290322580645])]
        #print(self.player_colour_list)




    def preprocessing(self):
        # adjust colour for player
        self.obs[np.where(self.obs==self.player_colour_list[self.player_1_nr])] = self.player_1_colour
        self.obs[np.where(self.obs==self.player_colour_list[self.player_2_nr])] = self.player_2_colour

        self.player_1_alive = True if self.player_1_colour in self.obs else False
        self.player_2_alive = True if self.player_2_colour in self.obs else False


    def predict(self, obs, player_nr, train):
        self.obs = obs
        #print(player_nr)
        self.player_1_nr = player_nr[0]
        self.player_2_nr = player_nr[1]
        self.preprocessing()




        # predict 2 actions
        action_1 = np.random.choice(np.arange(4))
        action_2 = np.random.choice(np.arange(4))







        if train and (self.player_1_alive or self.player_2_alive):
            # append to buffer
            self.obs_memory.append(self.obs)
            self.action_memory.append([action_1, action_2+4])

        self.frames += 1
        #a_1 = np.argmax(action_1[:4])
        return [action_1, action_2]


    def train(self):
        print(self.X_train)
        print(self.y_train)
        input("sleepoiashdf")

        # load most recent weights
        #model = load_weights(self.model_name)


        # train network



        #save weights
        #model.save_weights(self.model_name)
