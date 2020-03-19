from ctypes import *
import numpy as np #I'll refer to numpy as np and tensorflow as tf
import struct
import tensorflow as tf
import h5py
import os
import random
import time
from memory_profiler import profile
import sys

SHRINK = 2 #how much to shrink the image cap

luft_util = CDLL("cpp/luft_util.dll") #load luft_util dll
getPix = luft_util.getPix
getPix.restype = c_ulonglong #set getpix return type to avoid seg faults

luft_util.init() #init, very important
pixLen = luft_util.getPLen() #get image data
img_h = luft_util.getH()
img_w = luft_util.getW()
nimg_h = int(img_h/SHRINK) #adjust data
nimg_w = int(img_w/SHRINK)