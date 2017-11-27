from ctypes import *
import numpy as np
import scipy.misc as smp
import struct
import PIL

ScoreArr = c_int32 * 2
score = ScoreArr(0, 0)

getMem = CDLL("getMem.dll")
getPix = CDLL("getPixels.dll")
getMem.init()
pixLen = getPix.init()
img_h = getPix.getH()
img_w = getPix.getW()

pixArr = c_char * pixLen
pix = pixArr()

getPix.getPix(byref(pix))

pixList = []

for h in range(img_h):
	pixList.append([])
	for w in range(img_w):
		i = img_h*h+w
		pixList[h].append(int((0.3*struct.unpack('B', pix[i])[0]) + (0.59*struct.unpack('B', pix[i+1])[0]) + (0.11*struct.unpack('B', pix[i+2])[0])))

pixList = np.array(pixList)

print(pixList)

img = PIL.Image.fromarray(pixList, mode="L")
img.show()

oScore = c_int32()
while True:
	getMem.readGameMem(byref(score))
	if oScore != score[0]:
		print("Score:", score[0], "Mult:", score[1])
		oScore = score[0]
getMem.closeP()

