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

getMem = CDLL("getMem.dll")
getPixels = CDLL("getPixels.dll")
getPix = getPixels.getPix
getPix.restype = c_ulonglong
getMem.init()
pixLen = getPixels.init()
img_h = getPixels.getH()
img_w = getPixels.getW()



oScore = c_int32()
while True:
	getMem.readGameMem(byref(score))
	if oScore != score[0]:
		print("Score:", score[0], "Mult:", score[1])
		oScore = score[0]
		printImg(pixLen, img_h, img_w)
getMem.closeP()

