from ctypes import *
import numpy as np
import scipy.misc as smp
import struct
import PIL

def printImg(plen, mh, mw):
	pix_ptr = cast(getPix(), POINTER(c_char))
	pixList = []
	#print("Width:", mw, "Height:", mh, "Total:", plen)
	for i in range(0, plen, 4):
		#pixList.append([struct.unpack('B', pix_ptr[i])[0], struct.unpack('B', pix_ptr[i+1])[0], struct.unpack('B', pix_ptr[i+2])[0]])
		pixList.append(int((0.3*struct.unpack('B', pix_ptr[i])[0]) + (0.59*struct.unpack('B', pix_ptr[i+1])[0]) + (0.11*struct.unpack('B', pix_ptr[i+2])[0])))
	pixArr = np.array(pixList, dtype=np.int8).reshape((mh, mw))
	#print(pixArr)
	img = PIL.Image.fromarray(pixArr, mode="L")
	img.show()

ScoreArr = c_int32 * 2
score = ScoreArr(0, 0)

luft_util = CDLL("luft_util.dll")
getPix = luft_util.getPix
getPix.restype = c_ulonglong
luft_util.init()
pixLen = luft_util.getPLen()
img_h = luft_util.getH()
img_w = luft_util.getW()



oScore = c_int32()
while True:
	luft_util.readGameMem(byref(score))
	if oScore != score[0]:
		print("Score:", score[0], "Mult:", score[1])
		oScore = score[0]
		printImg(pixLen, img_h, img_w)
luft_util.closePMem()

