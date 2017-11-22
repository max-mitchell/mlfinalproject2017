#include <iostream>
#include <windows.h>
#include <Tchar.h>
#include <chrono>
#include <ctime>

#pragma comment(lib, "user32.lib")
#pragma comment(lib, "gdi32.lib")

int main() {

	using namespace std::chrono;

	HWND window = FindWindowA(0, _T("LUFTRAUSERS")); 
    if( window == 0 ){ 
        printf("Window not found!\n"); 
        exit(1);
    } 

    high_resolution_clock::time_point t1 = high_resolution_clock::now();

    RECT rect;
    GetWindowRect(window, &rect);
    int MAX_WIDTH = rect.right - rect.left;
    int MAX_HEIGHT = rect.bottom - rect.top;

    HDC dc = GetDC(window);
    HDC dcTmp = CreateCompatibleDC(dc);

    int iBpi= GetDeviceCaps(dcTmp, BITSPIXEL);

    BITMAPINFO bitmap;
    bitmap.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
    bitmap.bmiHeader.biWidth = MAX_WIDTH;
    bitmap.bmiHeader.biHeight = MAX_HEIGHT;
    bitmap.bmiHeader.biPlanes = 1;
    bitmap.bmiHeader.biBitCount  = iBpi;
    bitmap.bmiHeader.biCompression = BI_RGB;

    BYTE *data;
    HBITMAP hBitmap = CreateDIBSection(dcTmp, &bitmap, DIB_RGB_COLORS, (void**)&data, NULL, NULL);
    SelectObject(dcTmp, hBitmap);

    BitBlt(dcTmp, 0, 0, MAX_WIDTH, MAX_HEIGHT, dc , 0, 0, SRCCOPY);

    for (int i = 0; i < MAX_WIDTH * MAX_HEIGHT * 3; i+=3) {
    //	printf("(%d, %d, %d)\n", data[i], data[i+1], data[i+2]);
	}

	high_resolution_clock::time_point t2 = high_resolution_clock::now();

	duration<double, std::milli> diff = t2 - t1;

	printf("Time: %lf\n", diff.count());

    exit(0);
}