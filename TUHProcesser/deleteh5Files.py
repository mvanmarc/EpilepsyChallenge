from loaders.utils_preprocessing import getSubFolders, getSubFiles
from pathlib import Path
import os
from tqdm import tqdm

data_path = Path('/esat/biomeddata/kkontras/TUH/tuh_eeg/tuh_eeg_seizure/v2.0.3/edf') # Path of data files
groupids = getSubFolders(data_path)
for grp in (groupids):
    print("Busy removing files in "+grp+" folder ...")
    sub_ids = getSubFolders(os.path.join(data_path, grp))
    for sub in tqdm(sub_ids):
        sessions = getSubFolders(os.path.join(data_path, grp,sub))
        for sess in sessions:
            monts = getSubFolders(os.path.join(data_path, grp,sub,sess))
            for mont in monts:
                recs = getSubFiles(os.path.join(data_path, grp,sub,sess, mont), ".h5")
                for rec in recs:
                    rec_path = os.path.join(data_path, grp,sub,sess, mont, rec)
                    try: 
                        os.remove(rec_path)
                        # print(f"File '{rec_path}' deleted successfully.")
                    except FileNotFoundError:
                        print(f"File '{rec_path}' not found.")
                    except Exception as exc:
                        print(exc)