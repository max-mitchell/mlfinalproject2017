from ctypes import *
import numpy as np #I'll refer to numpy as np and tensorflow as tf
import h5py
import struct
import os
import time

SHRINK = 2 #how much to shrink the image cap
DEAD_VAL = 81 #screen color when dead
IS_DEAD = False #if AI is dead

def getImgData(plen, nw, nh, plist, pspot): #gets pixel data from luft_util
	global IS_DEAD
	pix_ptr = cast(getPix(SHRINK), POINTER(c_char)) #actual method call

	for i in range(nh*nw): #unpack each pixel and add it to np arr
		plist[i][pspot] = struct.unpack('B', pix_ptr[i])[0]


	IS_DEAD = True
	for i in range(50,250,4): #check if screen is dead screen
		if plist[i^2][pspot] != DEAD_VAL:
			IS_DEAD = False

luft_util = CDLL("cpp/luft_util.dll") #load luft_util dll
getPix = luft_util.getPix
getPix.restype = c_ulonglong #set getpix return type to avoid seg faults

luft_util.init() #init, very important
pixLen = luft_util.getPLen() #get image data
img_h = luft_util.getH()
img_w = luft_util.getW()
nimg_h = int(img_h/SHRINK) #adjust data
nimg_w = int(img_w/SHRINK)

H5FILE = h5py.File("data/d_table.hdf5", "r+")
SCORE_AVG = H5FILE["avg"]
GAME_NUM = SCORE_AVG.attrs["game_num"]
print("Loaded hdf5 file ", "GAME_NUM:", SCORE_AVG.attrs["game_num"], " w:", nimg_w, " h:", nimg_h)

time.sleep(2)

luft_util.sendKey(2) #start first game
time.sleep(.1)
luft_util.sendKey(6)
time.sleep(.2)

luft_util.sendKey(0)

print("Starting loop")
while True:
	plist = np.zeros((nimg_h*nimg_w, 1), dtype=np.int8)
	getImgData(pixLen, nimg_w, nimg_h, plist, 0)

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
