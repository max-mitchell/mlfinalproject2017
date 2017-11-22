#include <iostream>
#include <stdlib.h>
#include "getMem.h"

#pragma comment(lib, "getMem.lib")

int main() { 

   init();

   DWORD *RTRN_V = (DWORD *)malloc(sizeof(DWORD)*2);
   RTRN_V[0] = 0;
   RTRN_V[1] = 1;
   DWORD oScore = 0;

   printf("Starting game, Score: 0, Mult: 0\n"); 

   int keepGoing = 1;

   while (keepGoing) {
      readGameMem(RTRN_V);
      
      if (oScore != RTRN_V[0]) {
         printf("Score: %d, Mult: %d\n", RTRN_V[0], RTRN_V[1]); 
         oScore = RTRN_V[0];
         if (RTRN_V[0] == 0) {
            printf("Game over\n");
         }
      }
   }

   closeP();
}