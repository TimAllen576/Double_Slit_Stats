"""Removes a known bad pixel and
separates out images with only one active pixel
"""

import numpy as np
import pandas as pd
import h5py
import os
import re

clean_data = None
hdf_folder = r"C:\Users\Timojhoe\Documents\Uni\Fourth year\Double_Slit_Stats\hdf_files"
hdf_files = [f for f in os.listdir(hdf_folder) if f.endswith(".hdf")]

for file in hdf_files:
    hdf5_file = h5py.File(hdf_folder + "\\" + file)
    data = hdf5_file["entry/data/data"]
    # print(data.dtype)
    for slice_num, imslice in enumerate(data):
        indices = np.argwhere(imslice == 1)
        if len(indices) == 1:
            index = indices[0][::-1]
            file_id = re.findall(r'\d+', file)
            info = np.hstack(
                (index, slice_num, int(file_id[0]), int(file_id[1])))
            if clean_data is None:
                clean_data = info
            else:
                clean_data = np.vstack((clean_data, info))
    hdf5_file.close()

pd.DataFrame(clean_data).to_csv("clean_data.csv", index=False, sep="\t",
                                header=["x", "y", "slice", "Y", "Z"])
