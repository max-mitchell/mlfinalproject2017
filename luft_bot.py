from ctypes import *
import numpy as np
import scipy.misc as smp
import struct
import tensorflow as tf

SHRINK = 2
DEAD = 81

D_TABLE = []

def getImgData(plen, nw, nh):
	pix_ptr = cast(getPix(SHRINK), POINTER(c_char))
	pixList = []
	for i in range(nh*nw):
		pixList.append(struct.unpack('B', pix_ptr[i])[0])

	isDead = True
	for i in pixList[30000:30100]:
		if i != DEAD:
			isDead = False

	return np.array(pixList).reshape([nw, nh, 1])

def filld(plen, nw, nh):
	for i in range(32):
		e = [[], i%8, 1, []]
		for j in range(4):
			e[0].append(getImgData(plen, nw, nh))
		e[3] = e[0][1:]
		e[3].append(getImgData(plen, nw, nh))
		D_TABLE.append(e)

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

def trainCNN(pdata, npdata, nw, nh, lrt):
	
	x = tf.placeholder(tf.float32, shape=[4, nw, nh, 1])
	y = tf.placeholder(tf.float32, [8])

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

	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=out_layer))
	trainer = tf.train.AdamOptimizer(lrt).minimize(cost)

	sess = tf.Session()
	init = tf.global_variables_initializer().run(session=sess)

	for i in range(16):
		_, pi = sess.run([trainer, out_layer], feed_dict={y: [0, 0, 0, 0, 0, 0, 1, 0], x: pdata})
	print(pi[0])

def runConvNet(plen, nw, nh, lrt):
	filld(plen, nw, nh)
	e = [[], 0, 0, []]
	pdata = []
	npdata = []
	for i in range(4):
		pdata.append(getImgData(plen, nw, nh))
	npdata = pdata[1:]
	npdata.append(getImgData(plen, nw, nh))
	e[0] = pdata
	e[3] = npdata
	trainCNN(pdata, npdata, nw, nh, lrt)


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

runConvNet(pixLen, nimg_w, nimg_h, l_rate)

luft_util.closePMem()

