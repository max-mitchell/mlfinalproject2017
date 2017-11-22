from ctypes import *

ScoreArr = c_int32 * 2
score = ScoreArr(0, 0)

getMem = CDLL("getMem.dll")
getPix = CDLL("getPixels.dll")
getMem.init()
pixLen = getPix.init()

pixArr = c_char * pixLen
pix = pixArr()

getPix.getPix(byref(pix))

pixList = []

for i in range(0, pixLen, 3):
	pixList.append([pix[i], pix[i+1], pix[i+2]])

print(pixList[2000:2010])

oScore = c_int32()
while True:
	getMem.readGameMem(byref(score))
	if oScore != score[0]:
		print("Score:", score[0], "Mult:", score[1])
		oScore = score[0]
getMem.closeP()

