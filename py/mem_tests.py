from ctypes import *
import numpy as np #I'll refer to numpy as np and tensorflow as tf
import struct
import tensorflow as tf
import h5py
import os
import random
import time
import datetime
from memory_profiler import profile
import sys


test_type = 2
SHRINK = 2
IS_DEAD = False

def getImgData(plen, nw, nh, plist, pspot): #gets pixel data from luft_util
	pix_ptr = cast(getPix(), POINTER(c_char)) #actual method call

	for i in range(nh*nw): #unpack each pixel and add it to np arr
		plist[i][pspot] = struct.unpack('B', pix_ptr[i])[0]


	for i in range(50,250,4): #check if screen is dead screen
		if plist[i^2][pspot] != 81:
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
def runConvNet(w_small, h_small, learn_rate): #the bulk of the python code
	program_start_time = datetime.datetime.now()


	image_layer = tf.placeholder(tf.float32, [1, w_small, h_small, 4]) #actual image

	last_action_layer = tf.placeholder(tf.float32, [8]) #stores which action was taken last time
	current_score_layer = tf.placeholder(tf.float32, [1]) #stores score at time of event
	last_four_screens_out = tf.placeholder(tf.float32, [8]) #stores out_final_layer from first four screens

	conv_1 = makeConvLayer(image_layer, 4, 32, [8, 8], [1, 1], [4, 4]) #run through conv net
	conv_2 = makeConvLayer(conv_1, 32, 64, [4, 4], [1, 1], [3, 3])
	conv_3 = makeConvLayer(conv_2, 64, 64, [3, 3], [1, 1], [1, 1])
	conv_out_len = int(512) #as long as Luftrausers is at its default resolution, this works

	conv_3_flat = tf.reshape(conv_3, [1, conv_out_len]) #stretch the output back out

	weight_dense = tf.Variable(tf.truncated_normal([conv_out_len, 512], stddev=0.03)) #run two fully connected layers
	bias_dense = tf.Variable(tf.truncated_normal([512], stddev=0.01))
	dense_out = tf.matmul(conv_3_flat, weight_dense) + bias_dense
	dense_out = tf.nn.relu(dense_out)

	weight_out = tf.Variable(tf.truncated_normal([512, 8], stddev=0.03))
	bias_out = tf.Variable(tf.truncated_normal([8], stddev=0.01))
	out_final_layer = tf.matmul(dense_out, weight_out) + bias_out
	out_final_layer = tf.nn.relu(out_final_layer) #get actions probs for the controlls

	cost = (current_score_layer * last_action_layer + last_four_screens_out * last_action_layer - np.amax(out_final_layer) * last_action_layer) ** 2 #make cost
	trainer = tf.train.AdamOptimizer(learn_rate).minimize(cost) #minimize cost

	saver = tf.train.Saver() #makes saver so we can save our net!

	session = tf.Session()
	init_Qnet = tf.global_variables_initializer().run(session=session)

	for i in range(20000):
		session.run(trainer, feed_dict={image_layer: [np.random.randint(1, size=(w_small, h_small, 4))], 
			current_score_layer: [np.random.randint(1)], 
			last_action_layer: [np.random.randint(1)] * 8, 
			last_four_screens_out: [np.random.randint(1)] * 8})
		print(i, end="\r")
		
	print(session.run(trainer, feed_dict={image_layer: [np.random.randint(1, size=(w_small, h_small, 4))], 
				current_score_layer: [np.random.randint(1)], 
				last_action_layer: [np.random.randint(1)] * 8, 
				last_four_screens_out: [np.random.randint(1)] * 8}))

if test_type == 1:
	runConvNet(300, 100, .005)

elif test_type == 2:
	luft_util = CDLL("cpp/luft_util.dll") #load luft_util dll
	getPix = luft_util.getPix
	getPix.restype = c_ulonglong #set getpix return type to avoid seg faults

	luft_util.init(SHRINK) #init, very important
	pixLen = luft_util.getPLen() #get image data
	img_h = luft_util.getH()
	img_w = luft_util.getW()
	nimg_h = int(img_h/SHRINK) #adjust data
	nimg_w = int(img_w/SHRINK)

	for i in range(20000):
		plist = np.zeros((nimg_h*nimg_w, 1), dtype=np.int8)
		getImgData(pixLen, nimg_w, nimg_h, plist, 0)
		print(i, end="\r")