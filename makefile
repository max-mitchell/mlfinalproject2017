all: mem pix

mem: getMem.cpp getMem.h
	cl /LD getMem.cpp

pix: getPixels.cpp getPixels.h
	cl /LD getPixels.cpp

test: getMemTest.cpp getPixelTest.cpp
	cl getMemTest.cpp
	cl getPixelTest.cpp