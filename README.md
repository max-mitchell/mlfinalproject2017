# LuftBot
## Learning Luftrausers

This program aims to learn the game Luftrausers (http://luftrausers.com/) using deep reinforcement learning.

***

To make the dll, run `nmake` on the 64x Visual Studio dev cmd (64 bit because TensorFlow needs 64 bit python). Once the dll is there, `luft_bot.py` can be run normally. You can download the Visual C++ Build Tools here http://landinghub.visualstudio.com/visual-cpp-build-tools. Also, this program is Windows only, and requires Luftrausers to be in the foreground while running.