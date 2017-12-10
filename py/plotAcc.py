import h5py
import numpy as np
import matplotlib.pyplot as plt

H5FILE = h5py.File("data/d_table.hdf5", "r+")
SCORE_AVG = H5FILE["avg"]
GAME_NUM = SCORE_AVG.attrs["game_num"]

stp = [i for i in range(GAME_NUM)]

plt.scatter(stp, SCORE_AVG[:GAME_NUM])
plt.show()