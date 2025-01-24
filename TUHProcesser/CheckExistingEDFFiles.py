from loaders.utils_preprocessing import getSubFolders, getSubFiles
from pathlib import Path
import os
from tqdm import tqdm
import csv

lst = []
cnt = 0
data_path = Path('/esat/biomeddata/kkontras/TUH/tuh_eeg/tuh_eeg_seizure/v2.0.3/edf') # Path of data files
out_path = Path('dataset_lists') # Path of data files
groupids = getSubFolders(data_path)
for grp in tqdm(groupids):
    sub_ids = getSubFolders(os.path.join(data_path, grp))
    for sub in tqdm(sub_ids):
        sessions = getSubFolders(os.path.join(data_path, grp,sub))
        for sess in sessions:
            monts = getSubFolders(os.path.join(data_path, grp,sub,sess))
            for mont in monts:
                recs = getSubFiles(os.path.join(data_path, grp,sub,sess, mont), ".edf")
                cnt += len(recs)
print(cnt)
                    