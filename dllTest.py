from ctypes import *
import numpy as np
import scipy.misc as smp
import struct
import PIL
import time

FIRE_KEY = 0
LEFT_KEY = 1
UP_KEY = 2
RIGHT_KEY = 3

DEAD = 81

def printImg(plen, mh, mw):
	shrink = 2
	nh = int(mh/shrink)
	nw = int(mw/shrink)
	pix_ptr = cast(getPix(shrink), POINTER(c_char))
	pixList = []
	for i in range(nh*nw):
		pixList.append(struct.unpack('B', pix_ptr[i])[0])
	#pixArr = np.array(pixList, dtype=np.int8).reshape(nh, nw)
	
	#img = PIL.Image.fromarray(pixArr, mode="L")
	#img.show()

	#print(pixList[20000:20010])
	isDead = True
	for i in pixList[30000:30100]:
		if i != DEAD:
			isDead = False

	return isDead

ScoreArr = c_int32 * 2
score = ScoreArr(0, 0)

luft_util = CDLL("luft_util.dll")
getPix = luft_util.getPix
getPix.restype = c_ulonglong
luft_util.init()
pixLen = luft_util.getPLen()
img_h = luft_util.getH()
img_w = luft_util.getW()


printImg(pixLen, img_h, img_w)


time.sleep(5)

oScore = c_int32()
oDead = False
nGame = True
while True:
	dd = printImg(pixLen, img_h, img_w)
	luft_util.readGameMem(byref(score))
	if oScore != score[0]:
		if nGame and score[0] != 0:
			nGame = False
		if nGame and score[0] == 0:
			luft_util.sendKey(UP_KEY)
			time.sleep(.1)
			luft_util.sendKey(UP_KEY+4)
			luft_util.sendKey(FIRE_KEY)
		if score[0] == 0 and not nGame:
			luft_util.sendKey(FIRE_KEY+4)
		print("Score:", score[0], "Mult:", score[1], "Error:", GetLastError())
		oScore = score[0]
		if oDead == True:
			oDead = False
			dd = False
			luft_util.sendKey(FIRE_KEY)
	if dd != oDead:
		print("You died!")
		luft_util.sendKey(UP_KEY)
		time.sleep(.1)
		luft_util.sendKey(UP_KEY+4)
		oDead = dd
		nGame == True
		
luft_util.closePMem()

