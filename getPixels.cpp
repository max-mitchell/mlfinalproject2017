#include <iostream>
#include <windows.h>
#include <Tchar.h>
#include "getPixels.h"

#pragma comment(lib, "user32.lib")
#pragma comment(lib, "gdi32.lib")

#define _GETPIX_H_

extern "C" {

    HWND WINDOW;
    int MAX_WIDTH;
    int MAX_HEIGHT;
    RECT rect;

    int init() {
        WINDOW = FindWindowA(0, _T("LUFTRAUSERS")); 
        if(WINDOW == 0 ){ 
            printf("Window not found!\n"); 
            exit(1);
        } 

        GetWindowRect(WINDOW, &rect);
        MAX_WIDTH = rect.right - rect.left;
        MAX_HEIGHT = rect.bottom - rect.top;

        return MAX_HEIGHT * MAX_WIDTH * 3;
    }

    void getPix(BYTE *data) {
        HDC dc = GetDC(WINDOW);
        HDC dcTmp = CreateCompatibleDC(dc);

        int iBpi= GetDeviceCaps(dcTmp, BITSPIXEL);

        BITMAPINFO bitmap;
        bitmap.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
        bitmap.bmiHeader.biWidth = MAX_WIDTH;
        bitmap.bmiHeader.biHeight = MAX_HEIGHT;
        bitmap.bmiHeader.biPlanes = 1;
        bitmap.bmiHeader.biBitCount  = iBpi;
        bitmap.bmiHeader.biCompression = BI_RGB;

        HBITMAP hBitmap = CreateDIBSection(dcTmp, &bitmap, DIB_RGB_COLORS, (void**)&data, NULL, NULL);
        SelectObject(dcTmp, hBitmap);
        BitBlt(dcTmp, 0, 0, MAX_WIDTH, MAX_HEIGHT, dc , 0, 0, SRCCOPY);
    }

    int main() {

    }
}