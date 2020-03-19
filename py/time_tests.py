from ctypes import *
import numpy as np #I'll refer to numpy as np and tensorflow as tf
import struct

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 

import h5py
import os
import random
import time
import datetime
import sys

SHRINK_VAL = 2 #how much to shrink the image cap
DEAD_VAL = 81 #screen color when dead
IS_DEAD = False #if AI is dead

h5py_init = False
random_training = True #IMPORTANT whether or not to create new d_table
export_conv_net = False

DTABLE_LEN = 1500000 #d_table len
DTABLE_SPOT = 0 #spot to replace when adding event to d_table

score_arr = c_int32 * 2
SCORE = score_arr(0, 0) #ctypes arr for score and mult variables

l_rate = .0001 #learning rate

rand_move_chance = .01

score_save_rate = 5

main_loop_count = 100000

key_code = ["FIRE  ON", "Left  ON", "Up  ON", "Right  ON", "FIRE Off", "Left Off", "Up Off", "Right Off"]

def getImgData(w_small, h_small, p_list, p_spot): #gets pixel data from luft_util
	global IS_DEAD
	pix_ptr = cast(get_pixels(), POINTER(c_char)) #actual method call

	for i in range(h_small*w_small): #unpack each pixel and add it to np arr
		p_list[i][p_spot] = struct.unpack('B', pix_ptr[i])[0]


	IS_DEAD = True
	for i in range(50,250,4): #check if screen is dead screen
		if p_list[i^2][p_spot] != DEAD_VAL:
			IS_DEAD = False



def getShortList(l_len): #gets l_len items from the d_table
	s_list = []
	for i in range(DTABLE_SPOT):
		if i < l_len:
			s_list.append(i)
		elif random.random() < l_len/(i+1):
			s_list[random.randint(0, len(s_list)-1)] = i
	return s_list

def newGame(rand_mode): #performs new game actions
	global IS_DEAD
	for i in range(4, 8):
		luft_util.sendKey(i)
	time.sleep(2)
	luft_util.sendKey(2)
	time.sleep(.1)
	luft_util.sendKey(6)
	time.sleep(1)
	luft_util.sendKey(2)
	time.sleep(.1)
	luft_util.sendKey(6)
	IS_DEAD = False

def makeConvLayer(p_data, channel_n, filter_n, filter_s, pool_s, stride_n): #make conv layer
    conv_filterS = [filter_s[0], filter_s[1], channel_n, filter_n] #set up filter to scan image with
    weights = tf.Variable(tf.truncated_normal(conv_filterS, stddev=0.03)) #set random weights to begin with
    bias = tf.Variable(tf.truncated_normal([filter_n])) #set random bias
    out_final_layer = tf.nn.conv2d(p_data, weights, [1, stride_n[0], stride_n[1], 1], padding='SAME') #make out_final_layer by scanning image with filter
    out_final_layer += bias 
    out_final_layer = tf.nn.relu(out_final_layer) #pass out_final_layer through relu
    k_size = [1, pool_s[0], pool_s[1], 1]
    strides = [1, 2, 2, 1]
    out_final_layer = tf.nn.max_pool(out_final_layer, ksize=k_size, strides=strides, padding='SAME') #finish conv layer with pool

    return out_final_layer
	
#@profile
def runConvNet(w_small, h_small, learn_rate, rand_mode): #the bulk of the python code
	global DTABLE_SPOT
	global rand_move_chance
	program_start_time = datetime.datetime.now()


	image_layer = tf.placeholder(tf.float32, [1, w_small, h_small, 4]) #actual image

	last_action_layer = tf.placeholder(tf.float32, [8]) #stores which action was taken last time
	current_score_layer = tf.placeholder(tf.float32, [1]) #stores score at time of event
	last_four_screens_out = tf.placeholder(tf.float32, [8]) #stores out_final_layer from first four screens

	conv_1 = makeConvLayer(image_layer, 4, 32, [8, 8], [1, 1], [4, 4]) #run through conv net
	conv_2 = makeConvLayer(conv_1, 32, 64, [4, 4], [1, 1], [3, 3])
	conv_3 = makeConvLayer(conv_2, 64, 64, [3, 3], [1, 1], [1, 1])
	conv_out_len = int(768) #as long as Luftrausers is at its default resolution, this works

	conv_3_flat = tf.reshape(conv_3, [1, conv_out_len]) #stretch the output back out

	weight_dense = tf.Variable(tf.truncated_normal([conv_out_len, 512], stddev=0.03)) #run two fully connected layers
	bias_dense = tf.Variable(tf.truncated_normal([512], stddev=0.01))
	dense_out = tf.matmul(conv_3_flat, weight_dense) + bias_dense
	dense_out = tf.nn.relu(dense_out)

	weight_out = tf.Variable(tf.truncated_normal([512, 8], stddev=0.03))
	bias_out = tf.Variable(tf.truncated_normal([8], stddev=0.01))
	out_final_layer = tf.matmul(dense_out, weight_out) + bias_out
	out_final_layer = tf.nn.relu(out_final_layer) #get actions probs for the controlls

	cost = (current_score_layer * last_action_layer + last_four_screens_out * last_action_layer - tf.reduce_max(out_final_layer) * last_action_layer) ** 2 #make cost
	trainer = tf.train.AdamOptimizer(learn_rate).minimize(cost) #minimize cost

	saver = tf.train.Saver() #makes saver so we can save our net!

	session = tf.Session()
	init_Qnet = tf.global_variables_initializer().run(session=session)

	current_dir = os.path.dirname(os.path.realpath(__file__))
	saver.restore(session, current_dir+"\..\data\luft.ckpt") #one time load of previous net
	print("Loaded tensorflow net")

	luft_util.sendKey(2) #start first game
	time.sleep(.1)
	luft_util.sendKey(6)
	time.sleep(.2)
	
	print("Starting loop")
	last_keypress_sent = -1
	last_score_recorded = 0
	total_summed_score = 0
	num_summed_games = 0
	num_games_stored = SCORE_AVG.attrs["game_num"]
	for i in range(main_loop_count): #LEARN THE GAME FOR A WHILE
		print("Step:", i)

		if not rand_mode: #if not making a new d_tablwxe, train

			t1 = time.time()
			for i in range(50):
				event_list = np.random.randint(0, high=DTABLE_SPOT, size=4) #gets 4 random events
			t2 = time.time()
			print((t2-t1)/50, " getShortList")
			print(event_list)

			for event in event_list: #train on every event in event_list
				t1 = time.time()
				for i in range(10):
					init_Qnet_out = session.run(out_final_layer, feed_dict={image_layer: [DTABLE_nextState[event]]})[0] #run init_Qnet screens through
				t2 = time.time()
				print((t2-t1)/10, " run pt1")

				init_Qnet_max = [0] * 8
				action_taken = DTABLE_action[event]
				init_Qnet_max[action_taken] = init_Qnet_out[action_taken] #make an array of all 0s except where the max of init_Qnet_out was
				action_suggested = [0] * 8
				action_suggested[action_taken] = 1 #make a last_action_layer of 0s except a 1 where the max of init_Qnet_out was
				
				t1 = time.time()
				for i in range(10):
					session.run(trainer, feed_dict={image_layer: [DTABLE_initState[event]], current_score_layer: [DTABLE_reward[event]], last_action_layer: action_suggested, last_four_screens_out: init_Qnet_max}) #finish training and update the weights
				t2 = time.time()
				print((t2-t1)/10, " run pt2")


		t1 = time.time()
		for i in range(50):
			new_event_screens_current = np.zeros((h_small*w_small, 4), dtype=np.int8) #after training, make a new event
		t2 = time.time()
		print((t2-t1)/50, " init zeros")

		new_event_reward = 0
		new_event_action = 0
		for i in range(4):
			getImgData(w_small, h_small, new_event_screens_current, i)
			if IS_DEAD: #check if the AI died
				break;
			time.sleep(.03)
		if not IS_DEAD:
			t1 = time.time()
			for i in range(50):
				new_event_screens_next = np.copy(new_event_screens_current)
			t2 = time.time()
			print((t2-t1)/50, " copy arr")

			t1 = time.time()
			for i in range(50):
				new_event_screens_current = np.reshape(new_event_screens_current, (w_small, h_small, 4))
			t2 = time.time()
			print((t2-t1)/50, " reshape arr")

			if not rand_mode: #not making d_table, set normally
				new_Qnet_out = session.run(out_final_layer, feed_dict={image_layer: [new_event_screens_current]})[0] #get suggested key for the current state
				if random.random() > rand_move_chance:
					new_keypress = new_Qnet_out.argmax()
					if new_Qnet_out[new_keypress] != 0 and new_keypress != last_keypress_sent:
						t1 = time.time()
						for i in range(50):
							luft_util.sendKey(int(new_keypress)) #send the chosen key stroke and see what happens
						t2 = time.time()
						print((t2-t1)/50, " send key")

						last_keypress_sent = new_keypress
				else:
					new_keypress = random.randint(0, 7)
					if new_keypress != last_keypress_sent:
						luft_util.sendKey(int(new_keypress)) #send the chosen key stroke and see what happens
						last_keypress_sent = new_keypress
					if rand_move_chance > .0001:
						rand_move_chance -= .001
			else: #else, set randomly
				new_keypress = random.randint(0, 7)
				if new_keypress != last_keypress_sent:
					luft_util.sendKey(int(new_keypress)) #send the chosen key stroke and see what happens
					last_keypress_sent = new_keypress

			t1 = time.time()
			for i in range(50):
				luft_util.readGameMem(byref(SCORE)) #read score
			t2 = time.time()
			print((t2-t1)/50, " read mem")

			getImgData(w_small, h_small, new_event_screens_next, 0)
			new_event_screens_next = np.reshape(new_event_screens_next, (w_small, h_small, 4))
			if IS_DEAD: #if AI died, don't add new event
				last_keypress_sent = -1
				if not rand_mode:
					num_summed_games += 1
					total_summed_score += SCORE[0]
					if num_summed_games % score_save_rate == 0:
						SCORE_AVG[num_games_stored] = total_summed_score/score_save_rate
						print("Average Score over last", score_save_rate, "games:", (total_summed_score/score_save_rate))
						SCORE_AVG.attrs["game_num"] += 1
						total_summed_score = 0
						num_games_stored += 1
						
					
				newGame(rand_mode)
			else: #else, add event to spot DTABLE_SPOT in d_table
				if last_score_recorded == SCORE[0]:
					new_event_reward = 0
				else:
					#new_event_reward = SCORE[0] - last_score_recorded
					new_event_reward = SCORE[1]
					if new_event_reward < 0:
						new_event_reward = 0
					last_score_recorded = SCORE[0]
				
				t1 = time.time()
				for i in range(50):
					new_event_action = new_keypress
					np.copyto(DTABLE_initState[DTABLE_SPOT], new_event_screens_current)
					np.copyto(DTABLE_nextState[DTABLE_SPOT], new_event_screens_next)
					DTABLE_reward[DTABLE_SPOT] = new_event_reward
					DTABLE_action[DTABLE_SPOT] = new_event_action
					
					DTABLE_SPOT = (DTABLE_SPOT + 1) % DTABLE_LEN #change DTABLE_SPOT
					DTABLE_initState.attrs["dspot"] = DTABLE_SPOT
				t2 = time.time()
				print((t2-t1)/50, " save event")

		else:
			last_keypress_sent = -1
			if not rand_mode:
				num_summed_games += 1
				total_summed_score += SCORE[0]
				if num_summed_games % score_save_rate == 0:
					SCORE_AVG[num_games_stored] = total_summed_score/score_save_rate
					print("Average Score over last", score_save_rate, "games:", (total_summed_score/score_save_rate))
					SCORE_AVG.attrs["game_num"] += 1
					total_summed_score = 0
					num_games_stored += 1
				
			newGame(rand_mode)
	program_running_time = datetime.datetime.now() - program_start_time
	print(program_running_time)

def renderConvNet():
	session = tf.Session()
	init_Qnet = tf.global_variables_initializer().run(session=session)

	current_dir = os.path.dirname(os.path.realpath(__file__))
	tf.train.Saver().restore(session, current_dir+"\..\data\luft.ckpt") #one time load of previous net
	print("Loaded tensorflow net")



luft_util = CDLL("cpp/luft_util.dll") #load luft_util dll
get_pixels = luft_util.getPix
get_pixels.restype = c_ulonglong #set getpix return type to avoid seg faults

luft_util.init(SHRINK_VAL) #init, very important
total_pixel_count = luft_util.getPLen() #get image data
img_h = luft_util.getH()
img_w = luft_util.getW()
img_h_small = int(img_h/SHRINK_VAL) #adjust data
img_w_small = int(img_w/SHRINK_VAL)

if export_conv_net:
	print("Exp Conv Net Init")
	renderConvNet()
	raise SystemExit

if h5py_init:
	H5FILE = h5py.File("data/d_table.hdf5", "w-")
	H5FILE.create_dataset("dtable/sinit", (DTABLE_LEN, img_w_small, img_h_small, 4), dtype="i8", chunks=True, compression="lzf", shuffle=True)
	H5FILE.create_dataset("dtable/snext", (DTABLE_LEN, img_w_small, img_h_small, 4), dtype="i8", chunks=True, compression="lzf", shuffle=True)
	H5FILE.create_dataset("dtable/r", (DTABLE_LEN,), dtype="i", chunks=True, compression="lzf", shuffle=True)
	H5FILE.create_dataset("dtable/a", (DTABLE_LEN,), dtype="i", chunks=True, compression="lzf", shuffle=True)
	H5FILE.create_dataset("avg", (2000000,), dtype="i", chunks=True, compression="lzf", shuffle=True)
	H5FILE["dtable"]["sinit"].attrs["dspot"] = 0
	H5FILE["avg"].attrs["game_num"] = 0
	raise SystemExit

H5FILE = h5py.File("data/d_table.hdf5", "r+")
DTABLE_initState = H5FILE["dtable"]["sinit"]
DTABLE_nextState = H5FILE["dtable"]["snext"]
DTABLE_reward = H5FILE["dtable"]["r"]
DTABLE_action = H5FILE["dtable"]["a"]
SCORE_AVG = H5FILE["avg"]
DTABLE_SPOT = DTABLE_initState.attrs["dspot"]
print("Loaded hdf5 file, DTABLE_SPOT:", DTABLE_SPOT, "GAME_NUM:", SCORE_AVG.attrs["game_num"])


runConvNet(img_w_small, img_h_small, l_rate, random_training) #start training

luft_util.closePMem() #free memory in luft_util
H5FILE.close()

