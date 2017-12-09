#ifndef _LUFTUTIL_H_
#define _LUFTUTIL_H_

#if defined _LUFTUTIL_H_
#define LUFTUTIL_EXPORT __declspec(dllexport)
#else
#define LUFTUTIL_EXPORT __declspec(dllimport)
#endif

#include <windows.h> 

extern "C" {
	LUFTUTIL_EXPORT void init(void);
	LUFTUTIL_EXPORT int getPLen(void);
	LUFTUTIL_EXPORT int getH(void);
	LUFTUTIL_EXPORT int getW(void);
	LUFTUTIL_EXPORT BYTE *getPix(int shrink);
	LUFTUTIL_EXPORT void readGameMem(int *rtrn);
	LUFTUTIL_EXPORT void sendKey(int action);
	LUFTUTIL_EXPORT void closePMem(void);
	
}

#endif