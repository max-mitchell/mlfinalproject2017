#include <windows.h> 
#include <TlHelp32.h> 
#include <iostream> 
#include <Tchar.h>
#include "getMem.h"

#pragma comment(lib, "user32.lib")

using namespace std; 

#define _GETMEM_H_

extern "C" {

   DWORD PID = 0; 
   int STATIC_OFFSET = 0x6aca90; 
   int SCORE_OFFSET = 0x138;
   int MULT_OFFSET = 0x128;

   BYTE *BASE_ADDR = 0; 

   HANDLE PROCESS;

   void init() {

      HWND window = FindWindowA(0, _T("LUFTRAUSERS")); 
      if( window == 0 ){ 
         printf("Window not found!\n"); 
         exit(1);
      } 

      
      GetWindowThreadProcessId(window, &PID); 
      PROCESS = OpenProcess(PROCESS_ALL_ACCESS, FALSE, PID); 
      HANDLE hSnapshot = CreateToolhelp32Snapshot(TH32CS_SNAPMODULE, PID); 
      MODULEENTRY32 ModuleEntry32 = {0}; 
      ModuleEntry32.dwSize = sizeof(MODULEENTRY32); 
      Module32First(hSnapshot, &ModuleEntry32);

      BASE_ADDR = ModuleEntry32.modBaseAddr;

      CloseHandle(hSnapshot);
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

   void closeP() {
      CloseHandle(PROCESS); 
      exit(0);
   }

   int main() {
   }
}