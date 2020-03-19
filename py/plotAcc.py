import h5py
import numpy as np
import matplotlib.pyplot as plt

H5FILE = h5py.File("data/d_table.hdf5", "r+")
SCORE_AVG = H5FILE["avg"]
GAME_NUM = SCORE_AVG.attrs["game_num"]

stp = [i for i in range(400)]
plt.scatter(stp, SCORE_AVG[:400])
plt.show()

file = open("d_out.csv", "w+")
for i in range(400):
	file.write(str(i)+","+str(SCORE_AVG[i])+"\n")
file.close()