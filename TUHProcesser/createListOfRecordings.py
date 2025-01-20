from pathlib import Path
import os
import csv
from tqdm import tqdm
from loaders.utils_preprocessing import getSubFolders, getSubFiles, isAnnotationAvailable

#########################################################
# Script to create the list of patients/recordings/edfs #
#########################################################

# Example of path: /esat/biomeddata/kkontras/TUH/tuh_eeg/tuh_eeg_seizure/v2.0.3/edf/train/aaaaaaac/s001_2002/02_tcp_le/aaaaaaac_s001_t000.edf

# Specify paths
out_path = Path('dataset_lists') # Path for output files
data_path = Path('/esat/biomeddata/kkontras/TUH/tuh_eeg/tuh_eeg_seizure/v2.0.3/edf/train') # Path of data files

# Construct and save list subject ids and recordings
with open(os.path.join(out_path, 'recordings.tsv'), 'w', newline='') as tsvfile_recs:
    writer_recs = csv.writer(tsvfile_recs, delimiter='\t', lineterminator='\n')
    writer_recs.writerow(["group_id", "subject_id", "session","montage", "recording", "path"])
    with open(os.path.join(out_path, 'subject_ids.tsv'), 'w', newline='') as tsvfile_subs:
        writer_subs = csv.writer(tsvfile_subs, delimiter='\t', lineterminator='\n')
        writer_subs.writerow(["subject_id"])

        # Get list of all patients
        sub_ids = getSubFolders(data_path)
        writer_subs.writerows([[x] for x in sub_ids])

        # Get list of all recordings
        print("Extract each recording for each out of", len(sub_ids),"patients:")
        for sub in tqdm(sub_ids):
            sessions = getSubFolders(os.path.join(data_path,sub))
            for sess in sessions:
                monts = getSubFolders(os.path.join(data_path,sub,sess))
                for mont in monts:
                    recs = getSubFiles(os.path.join(data_path,sub,sess, mont), ".edf")
                    for rec in recs:
                        rec_path = os.path.join(data_path,sub,sess, mont, rec)
                        writer_recs.writerow([sub, sess, mont, rec, rec_path])
                        if not isAnnotationAvailable(rec_path):
                            print("Annotation not found of recording:", rec_path)


# with h5py.File(os.path.join(config.save_dir, 'predictions', name, rec[0] + '_' + rec[1] + '_preds.h5'), 'w') as f:
#                 f.create_dataset('y_pred', data=y_pred)
#                 f.create_dataset('y_true', data=y_true)

#             gc.collect()