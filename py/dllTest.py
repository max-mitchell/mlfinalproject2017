from ctypes import *
import numpy as np
import struct
from PIL import Image
import time

FIRE_KEY = 0
LEFT_KEY = 1
UP_KEY = 2
RIGHT_KEY = 3

DEAD = 81

SHRINK = 2

def printImg(plen, mh, mw):
	nh = int(mh/SHRINK)
	nw = int(mw/SHRINK)
	pix_ptr = cast(getPix(), POINTER(c_char))
	pixList = np.zeros((nh*nw), dtype=np.int8)
	for i in range(nh*nw):
		pixList[i] = struct.unpack('B', pix_ptr[i])[0]
	pixArr = pixList.reshape(nh, nw)
	
	img = Image.fromarray(pixArr, mode="L")
	img.show()

	#print(pixList[20000:20010])
	isDead = True
	for i in pixList[30000:30100]:
		if i != DEAD:
			isDead = False

	return isDead

ScoreArr = c_int32 * 2
score = ScoreArr(0, 0)

luft_util = CDLL("cpp/luft_util.dll")
getPix = luft_util.getPix
getPix.restype = c_ulonglong
luft_util.init(SHRINK)
pixLen = luft_util.getPLen()
img_h = luft_util.getH()
img_w = luft_util.getW()


printImg(pixLen, img_h, img_w)


time.sleep(5)

oScore = c_int32()
oDead = False
nGame = True
for i in range(1000):
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


