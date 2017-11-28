#include <iostream>
#include <windows.h>
#include <TlHelp32.h> 
#include <Tchar.h>
#include "luft_util.h"

#pragma comment(lib, "user32.lib")
#pragma comment(lib, "gdi32.lib")

using namespace std; 

#define _LUFTUTIL_H_

extern "C" {

    HWND WINDOW;
    HANDLE PROCESS;

    int MAX_WIDTH;
    int MAX_HEIGHT;
    RECT rect;

    DWORD PID = 0; 
    int STATIC_OFFSET = 0x6aca90; 
    int SCORE_OFFSET = 0x138;
    int MULT_OFFSET = 0x128;

    BYTE *BASE_ADDR = 0; 

    void init() {
        WINDOW = FindWindowA(0, _T("LUFTRAUSERS")); 
        if(WINDOW == 0 ){ 
            printf("Window not found!\n"); 
            exit(1);
        } 

        GetWindowRect(WINDOW, &rect);
        MAX_WIDTH = rect.right - rect.left;
        MAX_HEIGHT = rect.bottom - rect.top;

      
        GetWindowThreadProcessId(WINDOW, &PID); 
        PROCESS = OpenProcess(PROCESS_ALL_ACCESS, FALSE, PID); 
        HANDLE hSnapshot = CreateToolhelp32Snapshot(TH32CS_SNAPMODULE, PID); 
        MODULEENTRY32 ModuleEntry32 = {0}; 
        ModuleEntry32.dwSize = sizeof(MODULEENTRY32); 
        Module32First(hSnapshot, &ModuleEntry32);

        BASE_ADDR = ModuleEntry32.modBaseAddr;

        CloseHandle(hSnapshot);
    }

    int getPLen() {
        return MAX_HEIGHT * MAX_WIDTH * 4;
    }

    int getH() {
        return MAX_HEIGHT;
    }

    int getW() {
        return MAX_WIDTH;
    }

    BYTE *getPix() {
        HDC dc = GetDC(WINDOW);
        HDC dcTmp = CreateCompatibleDC(dc);

        int iBpi= GetDeviceCaps(dcTmp, BITSPIXEL);
        BITMAPINFO bitmap;
        bitmap.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
        bitmap.bmiHeader.biWidth = MAX_WIDTH;
        bitmap.bmiHeader.biHeight = -MAX_HEIGHT;
        bitmap.bmiHeader.biPlanes = 1;
        bitmap.bmiHeader.biBitCount  = iBpi;
        bitmap.bmiHeader.biCompression = BI_RGB;

        BYTE *data;
        HBITMAP hBitmap = CreateDIBSection(dcTmp, &bitmap, DIB_RGB_COLORS, (void**)&data, NULL, NULL);
        SelectObject(dcTmp, hBitmap);
        BitBlt(dcTmp, 0, 0, MAX_WIDTH, MAX_HEIGHT, dc , 0, 0, SRCCOPY);
        return data;
    }

    void readGameMem(int *rtrn) {
        int score;
        int mult;
        int mem; 
        SIZE_T numBytesRead; 
        ReadProcessMemory(PROCESS, (LPCVOID)(BASE_ADDR+STATIC_OFFSET), &mem, sizeof(int), &numBytesRead); 
        ReadProcessMemory(PROCESS, (LPCVOID)(mem+SCORE_OFFSET), &score, sizeof(int), &numBytesRead); 
        ReadProcessMemory(PROCESS, (LPCVOID)(mem+MULT_OFFSET), &mult, sizeof(int), &numBytesRead); 

        rtrn[0] = score;
        rtrn[1] = mult;
    }

    void closePMem() {
        CloseHandle(PROCESS); 
    }

    int main() {

    }
}