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
	pix_ptr = cast(getPix(), POINTER(c_char))
	pixList = []
	#print("Width:", mw, "Height:", mh, "Total:", plen)
	for i in range(0, plen, 4):
		#pixList.append([struct.unpack('B', pix_ptr[i])[0], struct.unpack('B', pix_ptr[i+1])[0], struct.unpack('B', pix_ptr[i+2])[0]])
		pixList.append(int((0.3*struct.unpack('B', pix_ptr[i])[0]) + (0.59*struct.unpack('B', pix_ptr[i+1])[0]) + (0.11*struct.unpack('B', pix_ptr[i+2])[0])))
	#pixArr = np.array(pixList, dtype=np.int8).reshape((mh, mw))
	#print(pixArr)
	#img = PIL.Image.fromarray(pixArr, mode="L")
	#img.show()

	isDead = True
	for i in pixList[20000:20100]:
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

time.sleep(5)
luft_util.sendKey(FIRE_KEY)

oScore = c_int32()
oDead = False
nGame = False
while True:
	dd = printImg(pixLen, img_h, img_w)
	luft_util.readGameMem(byref(score))
	if oScore != score[0]:
		print("Score:", score[0], "Mult:", score[1])
		oScore = score[0]
		if oDead == True:
			oDead = False
			dd = False
			luft_util.sendKey(FIRE_KEY)
	if dd != oDead:
		print("You died!")
		luft_util.sendKey(UP_KEY)
		luft_util.sendKey(UP_KEY+4)
		luft_util.sendKey(UP_KEY)
		luft_util.sendKey(UP_KEY+4)
		oDead = dd
		
luft_util.closePMem()

