from ctypes import *
import numpy as np #I'll refer to numpy as np and tensorflow as tf
import h5py
import struct
import os
import time
from PIL import Image
import sys

SHRINK = 2 #how much to shrink the image cap
DEAD_VAL = 81 #screen color when dead
IS_DEAD = False #if AI is dead

def getImgData(frames): #gets pixel data from luft_util
	global IS_DEAD
	image_return = [] # array to return all frames as 1d list
	for i in range(frames):
		getPixNew(img_arr) #actual method call
		image_return.append(img_arr[:])

		IS_DEAD = True
		for j in range(50,250,4): #check if screen is dead screen
			if img_arr[j^2] != DEAD_VAL:
				IS_DEAD = False
		if IS_DEAD:
			return 0
		if i < frames - 1:
			time.sleep(0.06)
	return image_return



def getImgDataOld(plen, nw, nh, plist, pspot): #gets pixel data from luft_util
	global IS_DEAD
	pix_ptr = cast(getPix(), POINTER(c_char)) #actual method call

	for i in range(nh*nw): #unpack each pixel and add it to np arr
		plist[i][pspot] = struct.unpack('B', pix_ptr[i])[0]


	
	IS_DEAD = True
	for i in range(50,250,4): #check if screen is dead screen
		if plist[i^2][pspot] != DEAD_VAL:
			IS_DEAD = False


def getImgDataOldest(plen, nw, nh, plist, pspot): #gets pixel data from luft_util
	global IS_DEAD
	pix_ptr = cast(getPix(), POINTER(c_char)) #actual method call

	for i in range(nh*nw): #unpack each pixel and add it to np arr
		plist[i][pspot] = struct.unpack('B', pix_ptr[i])[0]


	
	IS_DEAD = True
	for i in plist[30000:30050][pspot]: #check if screen is dead screen
		if i != DEAD_VAL:
			IS_DEAD = False
	

luft_util = CDLL("cpp/luft_util.dll") #load luft_util dll
getPix = luft_util.getPix
getPix.restype = c_ulonglong #set getpix return type to avoid seg faults

getPixNew = luft_util.getPixNew
getPixNew.argtypes = [POINTER(c_int)]

print(getPixNew.argtypes)

luft_util.init(SHRINK) #init, very important
pixLen = luft_util.getPLen() #get image data
img_h = luft_util.getH()
img_w = luft_util.getW()
nimg_h = int(img_h/SHRINK) #adjust data
nimg_w = int(img_w/SHRINK)

img_arr = (c_int * (nimg_h * nimg_w))()

print(img_arr)


H5FILE = h5py.File("data/d_table.hdf5", "r+")
SCORE_AVG = H5FILE["avg"]
GAME_NUM = SCORE_AVG.attrs["game_num"]
print("Loaded hdf5 file ", "GAME_NUM:", SCORE_AVG.attrs["game_num"], 
				" w:", nimg_w, " h:", nimg_h)

time.sleep(2)

'''
luft_util.sendKey(2) #start first game
time.sleep(.1)
luft_util.sendKey(6)
time.sleep(.2)

luft_util.sendKey(0)
'''
arr = getImgData(4)
a0 = np.reshape(arr[0], (nimg_h, nimg_w))
a1 = np.reshape(arr[1], (nimg_h, nimg_w))
a2 = np.reshape(arr[2], (nimg_h, nimg_w))
a3 = np.reshape(arr[3], (nimg_h, nimg_w))
arr = np.dstack([np.reshape(arr[i], (nimg_h, nimg_w)) for i in range(4)])

Image.fromarray(arr[:,:,0], "L").show()
Image.fromarray(arr[:,:,1], "L").show()
Image.fromarray(arr[:,:,2], "L").show()
Image.fromarray(arr[:,:,3], "L").show()

print("Starting loop")
while False:

	
	t1 = time.time()
	for i in range(100):
		arr = getImgData(4)
		a0 = np.reshape(arr[0], (nimg_h, nimg_w))
		a1 = np.reshape(arr[1], (nimg_h, nimg_w))
		a2 = np.reshape(arr[2], (nimg_h, nimg_w))
		a3 = np.reshape(arr[3], (nimg_h, nimg_w))
		arr = np.dstack((a0,a1,a2,a3))
	t2 = time.time()
	print((t2 - t1)/100.0, " New")

	
	t1 = time.time()
	plist = np.zeros((nimg_h*nimg_w, 1), dtype=np.int8)
	for i in range(100):
		getImgDataOld(pixLen, nimg_w, nimg_h, plist, 0)
	t2 = time.time()
	print((t2 - t1)/100.0, " Old")

	'''
	if IS_DEAD:
		SCORE_AVG.attrs["game_num"] += 1
		print("You died ", SCORE_AVG.attrs["game_num"])

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

		luft_util.sendKey(0)
	'''