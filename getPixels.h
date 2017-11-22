#ifndef _GETPIX_H_
#define _GETPIX_H_

#if defined _GETPIX_H_
#define GETPIX_EXPORT __declspec(dllexport)
#else
#define GETPIX_EXPORT __declspec(dllimport)
#endif

#include <windows.h> 

extern "C" {
	GETPIX_EXPORT int init(void);
	GETPIX_EXPORT void getPix(BYTE *data);
}

#endif