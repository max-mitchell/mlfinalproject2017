from ctypes import *
import numpy as np
import scipy.misc as smp
import struct
import tensorflow as tf
import os
import random
import time

SHRINK = 2
DEAD_VAL = 81
IS_DEAD = False
DLEN = 600
OLD_D = DLEN - 1

makeDTable = True

dtmp = [[[], 0, 0, []]] * DLEN
D_TABLE = np.array(dtmp)
ScoreArr = c_int32 * 2
SCORE = ScoreArr(0, 0)


def getImgData(plen, nw, nh):
	global IS_DEAD
	pix_ptr = cast(getPix(SHRINK), POINTER(c_char))
	pixList = np.zeros((nh*nw), dtype=np.int8)
	for i in range(nh*nw):
		pixList[i] = struct.unpack('B', pix_ptr[i])[0]

	IS_DEAD = True
	for i in pixList[30000:30100]:
		if i != DEAD_VAL:
			IS_DEAD = False

	return pixList.reshape([nw, nh, 1])

def getShortList(llen):
	slist = []
	for i, e in enumerate(D_TABLE):
		if i < llen:
			slist.append(e)
		elif random.random() < llen/(i+1):
			slist[random.randint(0, len(slist)-1)] = e
	return slist

def newGame(rand):
	global IS_DEAD
	for i in range(4, 8):
		luft_util.sendKey(i)
	if not rand:
		time.sleep(2)
	else:
		time.sleep(8)
	luft_util.sendKey(2)
	time.sleep(.1)
	luft_util.sendKey(6)
	time.sleep(1)
	luft_util.sendKey(2)
	time.sleep(.1)
	luft_util.sendKey(6)
	IS_DEAD = False

def makeConvLayer(pData, channelN, filterN, filterS, poolS, strideN):
    conv_filterS = [filterS[0], filterS[1], channelN, filterN]
    weights = tf.Variable(tf.truncated_normal(conv_filterS, stddev=0.03))
    bias = tf.Variable(tf.truncated_normal([filterN]))
    out_layer = tf.nn.conv2d(pData, weights, [1, strideN[0], strideN[1], 1], padding='SAME')
    out_layer += bias
    out_layer = tf.nn.relu(out_layer)
    ksize = [1, poolS[0], poolS[1], 1]
    strides = [1, 2, 2, 1]
    out_layer = tf.nn.max_pool(out_layer, ksize=ksize, strides=strides, padding='SAME')

    return out_layer

def runConvNet(plen, nw, nh, lrt, rand):
	global OLD_D
	global D_TABLE
	x = tf.placeholder(tf.float32, [4, nw, nh, 1])

	mask = tf.placeholder(tf.float32, [8])
	rt = tf.placeholder(tf.float32, [1])
	oPred = tf.placeholder(tf.float32, [8])

	L1 = makeConvLayer(x, 1, 32, [8, 8], [1, 1], [4, 4])
	L2 = makeConvLayer(L1, 32, 64, [4, 4], [1, 1], [3, 3])
	L3 = makeConvLayer(L2, 64, 64, [3, 3], [1, 1], [1, 1])
	nlen = int(3072)

	L3_flat = tf.reshape(L3, [1, nlen])

	weight_d1 = tf.Variable(tf.truncated_normal([nlen, 512], stddev=0.03))
	bias_d1 = tf.Variable(tf.truncated_normal([512], stddev=0.01))
	denseL1 = tf.matmul(L3_flat, weight_d1) + bias_d1
	denseL1 = tf.nn.relu(denseL1)

	weight_out = tf.Variable(tf.truncated_normal([512, 8], stddev=0.03))
	bias_out = tf.Variable(tf.truncated_normal([8], stddev=0.01))
	out_layer = tf.matmul(denseL1, weight_out) + bias_out
	out_layer = tf.nn.relu(out_layer)

	cost = rt * mask + np.amax(out_layer) * mask - oPred
	trainer = tf.train.AdamOptimizer(lrt).minimize(cost)

	saver = tf.train.Saver()

	sess = tf.Session()
	init = tf.global_variables_initializer().run(session=sess)

	cdir = os.path.dirname(os.path.realpath(__file__))
	saver.restore(sess, cdir+"\luft.ckpt")
	print("Loaded tensorflow net")

	if not rand:
		D_TABLE = np.load("d_table.npy")
	
	for i in range(800):
		print("Step:", i, end="\r")

		if not rand:

			if i % 100 == 0:
				saver.save(sess, cdir+"\luft.ckpt")
				print("Saved net on step", i)

			elist = getShortList(4)

			for e in elist:
				pInit = sess.run(out_layer, feed_dict={x: e[0]})[0]
				qsa = [0] * 8
				qsa[e[2]] = pInit[e[2]]
				m = [0] * 8
				m[e[2]] = 1
				sess.run(trainer, feed_dict={x: e[3], rt: [e[1]], mask: m, oPred: qsa})

		elif i % 200 == 0 and i != 0:
			np.save("d_table.npy", D_TABLE)
			print("Saved d_table on step", i)
			luft_util.reset()

		en = [[], 0, 0, []]
		for i in range(4):
			en[0].append(getImgData(plen, nw, nh))
			if IS_DEAD:
				break
		if not IS_DEAD:
			en[3] = en[0][1:]
			pf = sess.run(out_layer, feed_dict={x: en[0]})[0]
			#print(pf)
			if not rand:
				ksend = pf.argmax()
			else:
				ksend = random.randint(0, 7)
			luft_util.sendKey(int(ksend))
			luft_util.readGameMem(byref(SCORE))
			en[3].append(getImgData(plen, nw, nh))
			if IS_DEAD:
				newGame(rand)
			else:
				en[1] = SCORE[0]
				en[2] = ksend

				D_TABLE[OLD_D][0] = en[0]
				D_TABLE[OLD_D][1] = en[1]
				D_TABLE[OLD_D][2] = en[2]
				D_TABLE[OLD_D][3] = en[3]
				OLD_D = (OLD_D - 1) % DLEN
		else:
			newGame(rand)


		

	np.save("d_table.npy", D_TABLE)
	saver.save(sess, cdir+"\luft.ckpt")



luft_util = CDLL("luft_util.dll")
getPix = luft_util.getPix
getPix.restype = c_ulonglong

ScoreArr = c_int32 * 2
score = ScoreArr(0, 0)

l_rate = .0001

luft_util.init()
pixLen = luft_util.getPLen()
img_h = luft_util.getH()
img_w = luft_util.getW()
nimg_h = int(img_h/SHRINK)
nimg_w = int(img_w/SHRINK)



luft_util.sendKey(2)
time.sleep(.1)
luft_util.sendKey(6)
time.sleep(.2)

runConvNet(pixLen, nimg_w, nimg_h, l_rate, makeDTable)

luft_util.closePMem()

