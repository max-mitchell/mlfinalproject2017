#ifndef _GETMEM_H_
#define _GETMEM_H_

#if defined _GETMEM_H_
#define GETMEM_EXPORT __declspec(dllexport)
#else
#define GETMEM_EXPORT __declspec(dllimport)
#endif

#include <windows.h> 

extern "C" {
	GETMEM_EXPORT void init(void);
	GETMEM_EXPORT void readGameMem(int *score);
	GETMEM_EXPORT void closeP(void);
}

#endif