from ctypes import *
import numpy as np #I'll refer to numpy as np and tensorflow as tf
import struct

import tensorflow as tf

import h5py
import os
import random
import time
import datetime
import sys


# from https://towardsdatascience.com/deep-reinforcement-learning-build-a-deep-q-network-dqn-to-play-cartpole-with-tensorflow-2-and-gym-8e105744b998
# and https://nbviewer.jupyter.org/github/fg91/Deep-Q-Learning/blob/master/DQN.ipynb

class ConvNetModel(tf.keras.Model):
    # Create network using Keras. Has an input layer, n hidden image layers, and an output layer
    # Uses relu for the image layers, as well as variance scaling
    def __init__(self, state_shape, num_actions, hidden=512):
        super(ConvNetModel, self).__init__()
        self.input_layer = tf.keras.layers.InputLayer(input_shape=state_shape)
        self.hidden_layers = []

        self.conv1 = tf.keras.layers.Conv2D(filters=32, kernel_size=8,
                            strides=4, kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2),
                            padding="valid", activation="relu", use_bias=False)

        self.conv2 = tf.keras.layers.Conv2D(filters=64, kernel_size=4,
                            strides=2, kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2),
                            padding="valid", activation="relu", use_bias=False)

        self.conv3 = tf.keras.layers.Conv2D(filters=64, kernel_size=3,
                            strides=1, kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2),
                            padding="valid", activation="relu", use_bias=False)
        
        self.conv4 = tf.keras.layers.Conv2D(filters=hidden, kernel_size=7,
                            strides=1, kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2),
                            padding="valid", activation="relu", use_bias=False)

        self.output_layer = tf.keras.layers.Dense(
            num_actions, kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0))
        
    @tf.function
    def call(self, inputs):
        z = self.input_layer(inputs)
        z = self.conv1(z)
        z = self.conv2(z)
        z = self.conv3(z)
        z = self.conv4(z)
        output = self.output_layer(z)
        return output


class GameLearner(object):
    def __init__(self, state_shape, num_actions, gamma, dtable,
                    max_experiences, min_experiences, prev_events_num, learning_rate):
        # Number of moves the learner can make
        self.num_actions = num_actions
        # Number of events to look at while training
        self.prev_events_num = prev_events_num
        self.optimizer = tf.optimizers.Adam(learning_rate)
        # How the learner values current vs future rewards
        self.gamma = gamma
        self.model = ConvNetModel(state_shape, num_actions)
        # Length of dtable
        self.max_experiences = max_experiences
        # Number of random moves to perform before training
        self.min_experiences = min_experiences
        # List of initial states
        self.initial_state = dtable["dtable"]["sinit"]
        # List of next states
        self.next_state = dtable["dtable"]["snext"]
        # List of event rewards
        self.reward = dtable["dtable"]["r"]
        # List of event actions taken
        self.action = dtable["dtable"]["a"]
        # For keeping track of average score
        self.score_avg = dtable["avg"]
        # Current location in dtable
        self.dtable_spot = self.initial_state.attrs["dspot"]

    # Get q value from model
    def predict(self, inputs):
        return self.model(np.atleast_2d(inputs.astype('float32')))

    # Train on model
    @tf.function
    def train(self, TargetNet):
        # Don't train if not enough events have been seen
        if self.dtable_spot < self.min_experiences:
            return 0
        # Make a list of prev_events_num random events to train with
        events_list = np.random.randint(low=0, high=self.dtable_spot, size=self.prev_events_num)
        states_initial = np.asarray([self.initial_state[i] for i in events_list])
        actions = np.asarray([self.action[i] for i in events_list])
        rewards = np.asarray([self.reward[i] for i in events_list])
        states_next = np.asarray([self.next_state[i] for i in events_list])

        # Get q value from model
        value_next = np.max(TargetNet.predict(states_next), axis=1)
        # Compute actual predicted reward
        actual_values = rewards + self.gamma * value_next

        # Find action values and calculate loss
        with tf.GradientTape() as tape:
            selected_action_values = tf.math.reduce_sum(
                self.predict(states_initial) * tf.one_hot(actions, self.num_actions), axis=1)
            loss = tf.math.reduce_sum(tf.square(actual_values - selected_action_values))
        
            # Perform backprop using built in Keras model functions
            variables = self.model.trainable_variables
            gradients = tape.gradient(loss, variables)
            self.optimizer.apply_gradients(zip(gradients, variables))

    # Grab either a random action or a predicted action
    # based on epsilon
    def get_action(self, states, epsilon):
        if np.random.random_sample() < epsilon:
            return np.random.choice(self.num_actions)
        else:
            return np.argmax(self.predict(np.atleast_2d(states))[0])

    # Add experience to d5table
    def add_experience(self, s_init, r, a, s_next):
        np.copyto(self.initial_state[self.dtable_spot], s_init)
        self.reward[self.dtable_spot] = r
        self.action[self.dtable_spot] = a
        np.copyto(self.next_state[self.dtable_spot], s_next)
        self.dtable_spot = (self.dtable_spot + 1) % self.max_experiences
        self.initial_state.attrs["dspot"] = self.dtable_spot

    # Copy model data to make a target network.
    # This network is only used to predict value_next
    def copy_weights(self, TrainNet):
        variables1 = self.model.trainable_variables
        variables2 = TrainNet.model.trainable_variables
        for v1, v2 in zip(variables1, variables2):
            v1.assign(v2.numpy())

# Class for storing important game data
class Luftrauser(object):
    # Init
    def __init__(self, shrink_val=2, dead_val=81, game_speed=1, wait_btwn_frames=0.028,
                    h5py_init=False, dtable_length=1500000):
        # Image is shrunk by this factor
        self.shrink_val = shrink_val
        # Screen goes this color on player death
        self.dead_val = dead_val
        # Multiplier for speed dependant timers
        self.game_speed_mult = 1 / game_speed
        # Time to wait between frame capture to get distinct frames
        self.wait_btwn_frames = wait_btwn_frames
        # Make a new hdf5 file?
        self.h5py_init = h5py_init
        # Length of dtable
        self.dtable_length = dtable_length


        # Is the player dead?
        self.is_dead = False
        
        # What the different actions are
        self.key_code = ["FIRE  ON", "Left  ON", "Up  ON", "Right  ON", "FIRE Off", "Left Off", "Up Off", "Right Off"]

        # Load the luft_util dll file
        self.luft_util = CDLL("cpp/luft_util.dll")

        # Fetches the game screen
        self.get_pixels_new = self.luft_util.getPixNew
        # Make sure the argtype is correct
        self.get_pixels_new.argtypes = [POINTER(c_ubyte)]

        # Sets up the dll
        self.luft_util.init(shrink_val)
        # Fetch adjusted width and height of the game window
        self.img_h_small = int(self.luft_util.getH()/self.shrink_val)
        self.img_w_small = int(self.luft_util.getW()/self.shrink_val)

        # Initialize an array to hold pixel data from the dll
        self.game_image = (c_ubyte * (self.img_h_small * self.img_w_small))()
        # Initialize an array to hold score data from the dll
        self.score = (c_int32 * 2)()

        # Make a new dtable...
        if self.h5py_init:
            self.make_new_dtable(dtable_length, self.img_h_small, self.img_w_small)

        # Or load previous one
        self.dtable = h5py.File("data/d_table.hdf5", "r+")
        print("Loaded hdf5 file, DTABLE_SPOT:", self.dtable["dtable"]["sinit"].attrs["dspot"])

    # Free memory and close dtable
    def close(self):
        self.luft_util.closePMem()
        self.dtable.close()

    # Write new hdf5 file
    def make_new_dtable(self, length, h, w):
        H5FILE = h5py.File("data/d_table.hdf5", "w-")
        H5FILE.create_dataset("dtable/sinit", (length, h, w, 4), dtype="i8", chunks=True, compression="lzf", shuffle=True)
        H5FILE.create_dataset("dtable/snext", (length, h, w, 4), dtype="i8", chunks=True, compression="lzf", shuffle=True)
        H5FILE.create_dataset("dtable/r", (length,), dtype="i", chunks=True, compression="lzf", shuffle=True)
        H5FILE.create_dataset("dtable/a", (length,), dtype="i", chunks=True, compression="lzf", shuffle=True)
        H5FILE.create_dataset("avg", (length,), dtype="i", chunks=True, compression="lzf", shuffle=True)
        H5FILE["dtable"]["sinit"].attrs["dspot"] = 0
        H5FILE["avg"].attrs["game_num"] = 0
        raise SystemExit

    def restructure_image(self, arr, frames):
        return np.dstack([np.reshape(arr[i], (self.img_h_small, self.img_w_small)) for i in range(frames)])

    # Grabes frames number of frames of the game screen
    def get_frames(self, frames):
        # Adds all frames entries to image_return
        image_return = []
        for i in range(frames):
            # Call dll method
            self.get_pixels_new(self.game_image)
            image_return.append(self.game_image[:])

            self.is_dead = True
            # Check if screen is post-death
            for j in range(50,250,4):
                if self.game_image[j^2] != self.dead_val:
                    self.is_dead = False
            if self.is_dead:
                break
            if i < frames - 1:
                time.sleep(self.wait_btwn_frames)
        if self.is_dead:
            return 0
        return self.restructure_image(image_return, frames)

    # Grab score and puts it in self.score
    def get_score(self):
        self.luft_util.readGameMem(byref(self.score))
        return self.score[0]

    # Start first game
    def first_game(self):
        self.luft_util.sendKey(2)
        time.sleep(.1 * self.game_speed_mult)
        self.luft_util.sendKey(6)
        time.sleep(.2 * self.game_speed_mult)

    # Resets keys and starts new game with UP UP
    def new_game(self):
        for i in range(4, 8):
            self.luft_util.sendKey(i)
        time.sleep(2 * self.game_speed_mult)
        self.luft_util.sendKey(2)
        time.sleep(.1 * self.game_speed_mult)
        self.luft_util.sendKey(6)
        time.sleep(1 * self.game_speed_mult)
        self.luft_util.sendKey(2)
        time.sleep(.1 * self.game_speed_mult)
        self.luft_util.sendKey(6)
        self.is_dead = False

    # Play and learn Luftrausers
    def play_luft(self, TrainNet, TargetNet, epsilon, weight_copy_interval, num_frames):
        # Spot in loop
        loop_iter = 0
        # Keep track of keys so we don't send duplicate actions
        last_action = -1
        # Keep track of previous score
        prev_score = 0

        # Score variable
        reward = 0

        # Grab init screen
        frames_next = self.get_frames(num_frames)

        # Play through one game
        while not self.is_dead:
            # Get action to perform
            action = TrainNet.get_action([frames_next], epsilon)
            # Copy old new frames to new old frames
            frames_initial = frames_next

            # Make sure action is different than last time
            if action != last_action:
                # Send action to game
                self.luft_util.sendKey(int(action))
                last_action = action
            
            # Grab new frames
            frames_next = self.get_frames(num_frames)
            if self.is_dead:
                break
            # Grab new score
            reward = self.get_score() - prev_score
            prev_score = reward

            #if self.is_dead:
            #    # Remove the last few events
            #    continue
            TrainNet.add_experience(frames_initial, reward, action, frames_next)
            TrainNet.train(TargetNet)
            loop_iter += 1
            if loop_iter % weight_copy_interval == 0:
                TargetNet.copy_weights(TrainNet)

        return reward




def main():

    # luft = Luftrauser(h5py_init=True)

    # Set up variables
    luft = Luftrauser()
    gamma = 0.99
    weight_copy_interval = 25
    num_frames = 4
    state_shape = [luft.img_h_small, luft.img_w_small, num_frames]
    num_actions = len(luft.key_code)
    dtable = luft.dtable
    max_experiences = luft.dtable_length
    min_experiences = 30000
    prev_events_num = 8
    learning_rate = 0.001

    # Create file logger to track results
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = 'log/gamelearner' + current_time
    summary_writer = tf.summary.create_file_writer(log_dir)

    # Create both networks
    TrainNet = GameLearner(state_shape, num_actions,
                            gamma, dtable, max_experiences, min_experiences,
                            prev_events_num, learning_rate)

    TargetNet = GameLearner(state_shape, num_actions,
                            gamma, dtable, max_experiences, min_experiences,
                            prev_events_num, learning_rate)


    # How many games to play
    N = 50000
    total_rewards = np.empty(N)

    # Set up epsilon for moving from random moves
    # to predicted moves
    epsilon = 0.99
    decay = 0.9999
    min_epsilon = 0.1

    # Start first game
    luft.first_game()

    # Play games
    for n in range(N):
        # Find new epsilon
        epsilon = max(min_epsilon, epsilon * decay)
        # Play one game
        total_reward = luft.play_luft(TrainNet, TargetNet, epsilon, weight_copy_interval, num_frames)
        # Add game score to score list
        total_rewards[n] = total_reward
        # Find avg
        avg_rewards = total_rewards[max(0, n - 5):(n + 1)].mean()

        # Write to log
        with summary_writer.as_default():
            tf.summary.scalar('episode reward', total_reward, step=n)
            tf.summary.scalar('running avg reward(5)', avg_rewards, step=n)

        # Print every once in a while
        if n % 10 == 0:
            print("eppisode:", n, "eppisode reward:", total_reward, "eps:", epsilon, "avg reward (last 5):", avg_rewards)
        
        # Start new game
        luft.new_game()

    luft.close()



if __name__ == "__main__":
    main()