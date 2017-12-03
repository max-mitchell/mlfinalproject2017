all: util

util: luft_util.cpp luft_util.h
	cl /LD luft_util.cpp