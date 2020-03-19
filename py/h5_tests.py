import h5py
import numpy as np

make_table = False

if make_table:
	test_h5 = h5py.File("data/test.hdf5", "w-")
	test_h5.create_dataset("test_arr", (500000, 5, 5, 4), dtype="i8", chunks=True, compression="lzf", shuffle=True)
else:
	test_h5 = h5py.File("data/test.hdf5", "r+")
	test_arr = test_h5["test_arr"]


for i in range(20000):
	test_arr[i] = np.random.randint(1, size=(5, 5, 4))
	print(i, end="\r")