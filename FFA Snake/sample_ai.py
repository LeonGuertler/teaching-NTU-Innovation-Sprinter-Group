import numpy as np
import cv2, h5py
import matplotlib.pyplot as plt

import keras
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Flatten, Dense, Dropout




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

        self.player_1_nr = 0
        self.player_2_nr = 0


        self.X_train = []
        self.y_train = []


        self.obs_memory = []
        self.action_memory = []

        self.model_name = "TEAM_NAME.model"


        self.threshold = 100

        self.trained = False



        self.model = Sequential()
        self.model.add(Conv2D(16, (3,3), activation="tanh", input_shape=(x_size,y_size,1)))
        self.model.add(MaxPooling2D())
        self.model.add(BatchNormalization())

        self.model.add(Conv2D(16, (3,3), activation="tanh"))
        self.model.add(MaxPooling2D())
        self.model.add(BatchNormalization())

        self.model.add(Conv2D(16, (5,5), activation="tanh"))
        self.model.add(MaxPooling2D())
        self.model.add(BatchNormalization())

        self.model.add(Flatten())

        self.model.add(Dense(512, activation="relu"))
        self.model.add(Dropout(.2))

        self.model.add(Dense(256, activation="relu"))
        self.model.add(Dropout(.2))

        self.model.add(Dense(256, activation="relu"))
        self.model.add(Dropout(.2))

        self.model.add(Dense(128, activation="relu"))

        self.model.add(Dense(8, activation="softmax"))

        self.model.compile(loss="mse", optimizer="adam", metrics=["accuracy"])



        self.TRAIN_LENGTH = 10_000#10_000

    def reset(self, train, test_obs):
        #print(f"{self.player_1_nr} & {self.player_2_nr}:\t{self.frames}")

        if self.frames>0 and train:
            #print(f"\t: {self.frames}")
            #print("\t"*(5-self.player_1_nr) + f"{self.frames}",end='\r')
            # append to training set (or not, kinda up to you man, don't @ us)
            print(self.frames, self.threshold)
            if self.frames > self.threshold:
                print("okasldkfh")
                for obs,a in zip(self.obs_memory, self.action_memory):
                    self.X_train.append(obs)
                    label = np.zeros((8))
                    label[a] = 1#self.frames
                    self.y_train.append(label)

        self.obs_memory = []
        self.action_memory = []

        self.frames = 0
        self.player_1_alive = True
        self.player_2_alive = True


        # body = 0
        # bg = 1
        # food = 0.516
        #print(np.unique(test_obs))
        self.player_colour_list = [e for e in np.unique(test_obs) if not (e in [0, 1, 0.5161290322580645])]
        #self.player_1_colour = self.player_colour_list[self.player_1_nr]
        #self.player_2_colour = self.player_colour_list[self.player_2_nr]
        #print(self.player_colour_list)


        # load most recent weights
        try:
            self.model = load_model(self.model_name)
        except:
            pass




    def preprocessing(self):
        # adjust colour for player
        self.obs[np.where(self.obs==self.player_colour_list[self.player_1_nr])] = .3#self.player_1_colour
        self.obs[np.where(self.obs==self.player_colour_list[self.player_2_nr])] = .4#self.player_2_colour

        print(np.unique(self.obs))

        self.player_1_alive = True if .3 in self.obs else False
        self.player_2_alive = True if .4 in self.obs else False


    def predict(self, obs, player_nr, train):
        self.obs = obs
        #print(np.shape(self.obs))
        #print(player_nr)
        self.player_1_nr = player_nr[0]
        self.player_2_nr = player_nr[1]
        self.preprocessing()
        self.obs = self.obs[..., np.newaxis]




        # predict 2 actions
        if np.random.uniform() < .02 or not self.trained:
            action_1 = np.random.choice(np.arange(4))
            action_2 = np.random.choice(np.arange(4))
        else:
            #print("random action")
            action = self.model.predict(np.asarray([self.obs]))[0]
            #print(f"action: {action}")
            action_1 = np.argmax(action[:4])
            action_2 = np.argmax(action[4:])


        if train and (self.player_1_alive or self.player_2_alive):
            # append to buffer
            self.obs_memory.append(self.obs)
            self.action_memory.append([action_1, action_2+4])

            self.frames += 1
        else:
            print(self.player_1_alive)
            print(self.player_2_alive)
            plt.imshow(obs)
            plt.show()
            print("oshdfjuklahsdfjklhasdjklfh")
        #a_1 = np.argmax(action_1[:4])
        return [action_1, action_2]


    def train(self):
        #print("start training")
        #print(f"{self.player_1_nr}: {len(self.)}")
        #print(self.X_train)
        #print(self.y_train)
        #input("sleepoiashdf")

        if len(self.y_train) >= self.TRAIN_LENGTH:

            # load most recent weights
            try:
                self.model = load_model(self.model_name)
            except:
                pass


            # train network
            self.model.fit(np.asarray(self.X_train), np.asarray(self.y_train), epochs=7)



            #save weights
            self.model.save_weights(self.model_name)

            self.X_train = []
            self.y_train = []

            self.trained = True
        else:
            print("\t", str(len(self.y_train)))
