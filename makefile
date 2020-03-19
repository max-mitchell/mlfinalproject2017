all: util

util: cpp\luft_util.cpp cpp\luft_util.h
	cl /Focpp\luft_util.obj /LD cpp\luft_util.cpp /link /out:cpp\luft_util.dll cpp\luft_util.exp cpp\luft_util.lib