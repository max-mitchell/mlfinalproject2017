#include <iostream>
#include <windows.h>
#include <TlHelp32.h> 
#include <Tchar.h>
#include "luft_util.h"

#pragma comment(lib, "user32.lib")
#pragma comment(lib, "gdi32.lib") //this is the screen cap library for windows

using namespace std; 

#define _LUFTUTIL_H_ //from .h file

extern "C" { //means it's good for any version of c 

    HWND WINDOW; //handle to the Luftrausers window
    HANDLE PROCESS; //handle to the Luftrausers process

    WORD FIRE_KEY = 0x58; //key codes for X,
    WORD LEFT_KEY = 0x41; //A
    WORD UP_KEY = 0x57; //W
    WORD RIGHT_KEY = 0x44; //D

    int MAX_WIDTH; //image w,h
    int MAX_HEIGHT;
    RECT rect; //rect for finding w,h

    DWORD PID = 0; //process id
    int STATIC_OFFSET = 0x6aca90; //offset from Luftrausers base mem address
    int SCORE_OFFSET = 0x138; //offset from static to score
    int MULT_OFFSET = 0x128; //offset from static to mult

    BYTE *BASE_ADDR = 0; //pointer to base addr

    int PIX_DEPTH = 4; //4 becuase Luftrausers uses RGBA coloring

    BYTE *NDATA; //array to hold processed pixel values

    int SHRINK = 2; //global shrink value

    void init(int shrink) { //init function
        WINDOW = FindWindowA(0, _T("LUFTRAUSERS")); //get window handle
        if(WINDOW == 0 ){ 
            printf("Window not found!\n"); 
            exit(1);
        } 

        GetWindowRect(WINDOW, &rect); //get image data
        MAX_WIDTH = rect.right - rect.left;
        MAX_HEIGHT = rect.bottom - rect.top;
      
        GetWindowThreadProcessId(WINDOW, &PID); //get pid 
        PROCESS = OpenProcess(PROCESS_ALL_ACCESS, FALSE, PID); //get process handle
        if(PROCESS == 0 ){ 
            printf("Process not found!\n"); 
            exit(1);
        } 
        HANDLE hSnapshot = CreateToolhelp32Snapshot(TH32CS_SNAPMODULE, PID); //get process snapshot
        MODULEENTRY32 ModuleEntry32 = {0}; 
        ModuleEntry32.dwSize = sizeof(MODULEENTRY32); 
        Module32First(hSnapshot, &ModuleEntry32);

        BASE_ADDR = ModuleEntry32.modBaseAddr; //find base addr from snapshot

        CloseHandle(hSnapshot); //free snapshot

        SHRINK = shrink;

        int nw = MAX_WIDTH/SHRINK;
        int nh = MAX_HEIGHT/SHRINK;
        NDATA = new BYTE[nw*nh]; //set up global pixel array
    }

    int getPLen() { //return len of full pixel array
        return MAX_HEIGHT * MAX_WIDTH * PIX_DEPTH;
    }

    int getH() { //return image height
        return MAX_HEIGHT;
    }

    int getW() { //return image height
        return MAX_WIDTH;
    }

    BYTE *getPix() { //get pixels from Luftrausers screen
        HDC dc = GetDC(WINDOW); //get dc object
        HDC dcTmp = CreateCompatibleDC(dc); //not sure what this line does
        

        int iBpi= GetDeviceCaps(dcTmp, BITSPIXEL); //get pixel depth, I hardcoded in a 4 but really I shouldn't have
        BITMAPINFO bitmap;
        bitmap.bmiHeader.biSize = sizeof(BITMAPINFOHEADER); //give the bitmap the specs of our image
        bitmap.bmiHeader.biWidth = MAX_WIDTH;
        bitmap.bmiHeader.biHeight = -MAX_HEIGHT;
        bitmap.bmiHeader.biPlanes = 1;
        bitmap.bmiHeader.biBitCount  = iBpi;
        bitmap.bmiHeader.biCompression = BI_RGB;


        BYTE *data = NULL;
        HBITMAP hBitmap = CreateDIBSection(dcTmp, &bitmap, DIB_RGB_COLORS, (void**)&data, NULL, NULL); //set up pixel capture
        HGDIOBJ pbitmap = SelectObject(dcTmp, hBitmap);
        BitBlt(dcTmp, 0, 0, MAX_WIDTH, MAX_HEIGHT, dc , 0, 0, SRCCOPY); //get pixel data
        if (data == NULL) {
            printf("Get Pixels Error: %d\n", GetLastError());
            exit(1);
        }

        int nw = MAX_WIDTH/SHRINK;
        int nh = MAX_HEIGHT/SHRINK;
        int g = 0;
        for (int i = 0; i < MAX_WIDTH*MAX_HEIGHT*PIX_DEPTH/SHRINK; i += PIX_DEPTH) { //convert RGBA values to grayscale, and limit length to nw*nh
            if (g >= nw*nh) {
                break;
            }
            NDATA[g] = 0.3*data[i*SHRINK] + 0.59*data[i*SHRINK+1] + 0.11*data[i*SHRINK+2];
            g++;
            if ((i/PIX_DEPTH) % MAX_WIDTH == 0) {
                i += MAX_WIDTH * PIX_DEPTH * (SHRINK - 1);
            }
        }
	    //InvalidateRect(WINDOW, &rect, 1);
      	ReleaseDC(WINDOW, dc); //free dc
        SelectObject(dcTmp, pbitmap);
        DeleteDC(dcTmp);
        DeleteObject(pbitmap);
        DeleteObject(hBitmap);
        return NDATA;
    }

    void getPixNew(BYTE *ndata) { //get pixels from Luftrausers screen
        HDC dc = GetDC(WINDOW); //get dc object
        HDC dcTmp = CreateCompatibleDC(dc); //not sure what this line does
        

        int iBpi= GetDeviceCaps(dcTmp, BITSPIXEL); //get pixel depth, I hardcoded in a 4 but really I shouldn't have
        BITMAPINFO bitmap;
        bitmap.bmiHeader.biSize = sizeof(BITMAPINFOHEADER); //give the bitmap the specs of our image
        bitmap.bmiHeader.biWidth = MAX_WIDTH;
        bitmap.bmiHeader.biHeight = -MAX_HEIGHT;
        bitmap.bmiHeader.biPlanes = 1;
        bitmap.bmiHeader.biBitCount  = iBpi;
        bitmap.bmiHeader.biCompression = BI_RGB;


        BYTE *data = NULL;
        HBITMAP hBitmap = CreateDIBSection(dcTmp, &bitmap, DIB_RGB_COLORS, (void**)&data, NULL, NULL); //set up pixel capture
        HGDIOBJ pbitmap = SelectObject(dcTmp, hBitmap);
        BitBlt(dcTmp, 0, 0, MAX_WIDTH, MAX_HEIGHT, dc , 0, 0, SRCCOPY); //get pixel data
        if (data == NULL) {
            printf("Get Pixels Error: %d\n", GetLastError());
            exit(1);
        }

        int nw = MAX_WIDTH/SHRINK;
        int nh = MAX_HEIGHT/SHRINK;
        int g = 0;
        for (int i = 0; i < MAX_WIDTH*MAX_HEIGHT*PIX_DEPTH/SHRINK; i += PIX_DEPTH) { //convert RGBA values to grayscale, and limit length to nw*nh
            if (g >= nw*nh) {
                break;
            }
            ndata[g] = 0.3*data[i*SHRINK] + 0.59*data[i*SHRINK+1] + 0.11*data[i*SHRINK+2];
            g++;
            if ((i/PIX_DEPTH) % MAX_WIDTH == 0) {
                i += MAX_WIDTH * PIX_DEPTH * (SHRINK - 1);
            }
        }
	    //InvalidateRect(WINDOW, &rect, 1);
      	ReleaseDC(WINDOW, dc); //free dc
        SelectObject(dcTmp, pbitmap);
        DeleteDC(dcTmp);
        DeleteObject(pbitmap);
        DeleteObject(hBitmap);
    }

    void readGameMem(int *rtrn) { //get score and mult variables
        int score;
        int mult;
        int mem; 
        SIZE_T numBytesRead; 
        if (!ReadProcessMemory(PROCESS, (LPCVOID)(BASE_ADDR+STATIC_OFFSET), &mem, sizeof(int), &numBytesRead)) { //first get offset address
            printf("Getting mem, error %d\n", GetLastError());
            exit(1);
        } 
        if (!ReadProcessMemory(PROCESS, (LPCVOID)(mem+SCORE_OFFSET), &score, sizeof(int), &numBytesRead)) { //read score
            printf("Getting mem, error %d\n", GetLastError());
            exit(1);
        }
        if (!ReadProcessMemory(PROCESS, (LPCVOID)(mem+MULT_OFFSET), &mult, sizeof(int), &numBytesRead)) { // read mult
            printf("Getting mem, error %d\n", GetLastError());
            exit(1);
        }

        rtrn[0] = score; //no nead to return, just pass the values to rtrn
        rtrn[1] = mult;
    }

    void sendKey(int action) { //send key
        if (action < 4) { //if <4, it's a press
            if (action == 0) {
                uint32_t lparam = 1 | (MapVirtualKey(FIRE_KEY, 0) << 16);
                SendMessage(WINDOW, WM_KEYDOWN, FIRE_KEY, lparam);
            } else if (action == 1) {
                uint32_t lparam = 1 | (MapVirtualKey(LEFT_KEY, 0) << 16);
                SendMessage(WINDOW, WM_KEYDOWN, LEFT_KEY, lparam);
            } else if (action == 2) {
                uint32_t lparam = 1 | (MapVirtualKey(UP_KEY, 0) << 16);
                SendMessage(WINDOW, WM_KEYDOWN, UP_KEY, lparam);
            } else if (action == 3) {
                uint32_t lparam = 1 | (MapVirtualKey(RIGHT_KEY, 0) << 16);
                SendMessage(WINDOW, WM_KEYDOWN, RIGHT_KEY, lparam);
            }
        } else { //if >=4, it's a release
            if (action == 4) {
                uint32_t lparam = 1 | (MapVirtualKey(FIRE_KEY, 0) << 16) | 0xC00000000;
                SendMessage(WINDOW, WM_KEYUP, FIRE_KEY, lparam);
            } else if (action == 5) {
                uint32_t lparam = 1 | (MapVirtualKey(LEFT_KEY, 0) << 16) | 0xC00000000;
                SendMessage(WINDOW, WM_KEYUP, LEFT_KEY, lparam);
            } else if (action == 6) {
                uint32_t lparam = 1 | (MapVirtualKey(UP_KEY, 0) << 16) | 0xC00000000;
                SendMessage(WINDOW, WM_KEYUP, UP_KEY, lparam);
            } else if (action == 7) {
                uint32_t lparam = 1 | (MapVirtualKey(RIGHT_KEY, 0) << 16) | 0xC00000000;
                SendMessage(WINDOW, WM_KEYUP, RIGHT_KEY, lparam);
            }
        }
        //printf("Sent key %d with error %d\n", action, GetLastError());
    }

    void closePMem() { //frees some memory
        CloseHandle(PROCESS); 
        delete [] NDATA;
    }

    int main() { //for testing purposes
        init(2);
        /*for (int i = 0; i < 5000; i++) {
            int *score = new int[2];
            readGameMem(score);
            delete [] score;
        }*/
        for (int i = 0; i < 10000; i++) {
            getPix();
        }
        /*for (int i = 0; i < 5000; i++) {
            sendKey(0);
            sendKey(4);
        }*/
        
        closePMem();
        exit(0);
    }

}