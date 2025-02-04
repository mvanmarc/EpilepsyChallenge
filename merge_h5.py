#I have in a folder a lot of h5 files that I want to merge into one h5 file.
#The h5 should have inside the following order: patient_id, recording_id, and then in there the data.

import h5py
import os
import numpy as np

# Get all h5 files from the tsv file
path = "./TUHProcesser/dataset_lists/existingh5.tsv"
with open(path, 'r') as f:
    lines = f.readlines()
    h5_files = [line.split('\t')[-1].strip() for line in lines]

# Create the new h5 file
new_h5 = h5py.File("/esat/biomeddata/kkontras/TUH/tuh_eeg/tuh_eeg_seizure/v2.0.3/TUH.h5", "w")

# Iterate over all h5 files
for h5_file in h5_files:
    with h5py.File(h5_file, "r") as f:
        patient_id = h5_file.split("/")[-1].split("_")[0]
        recording_id = h5_file.split("/")[-1].split("_")[1]
        data = f["data"][:]
        new_h5.create_dataset(f"{patient_id}/{recording_id}", data=data)



