from pathlib import Path
import os
import csv
from tqdm import tqdm

#########################################################
# Script to create the list of patients/recordings/edfs #
#########################################################

# Example of path: /esat/biomeddata/kkontras/TUH/tuh_eeg/tuh_eeg/v2.0.1/edf/000/aaaaaaaa/s001_2015/01_tcp_ar/aaaaaaaa_s001_t000.edf

def getSubFolders(path: Path) -> list:
    """ Get list of all subfolders in a given folder """
    try:
        res = [name for name in os.listdir(path) if os.path.isdir(os.path.join(path,name))]
        return sorted(res)
    except Exception as exc:
        print(exc)
        return []

def getSubEDFFiles(path: Path) -> list:
    """ Get list of all EDF files (.edf) in a given folder """
    try:
        res = [name for name in os.listdir(path) if os.path.isfile(os.path.join(path,name)) and name[-4:] == ".edf"]
        return sorted(res)
    except Exception as exc:
        print(exc)
        return []

# Specify paths
out_path = Path('/users/sista/mvanmarc/Documents/Doctoraat/12. Python/Challenge16Feb2025/TUHProcesser/dataset_lists') # Path for output files
data_path = Path('/esat/biomeddata/kkontras/TUH/tuh_eeg/tuh_eeg/v2.0.1/edf/') # Path of data files

# Construct and save list of patient groups, subject ids and recordings
sub_grp = getSubFolders(data_path)
with open(os.path.join(out_path, 'groups.tsv'), 'w', newline='') as tsvfile_grps:
    writer_grps = csv.writer(tsvfile_grps, delimiter='\t', lineterminator='\n')
    writer_grps.writerow(["group_id", "List of patient ids"])
    with open(os.path.join(out_path, 'recordings.tsv'), 'w', newline='') as tsvfile_recs:
        writer_recs = csv.writer(tsvfile_recs, delimiter='\t', lineterminator='\n')
        writer_recs.writerow(["group_id", "subject_id", "session","montage", "recording", "path"])
        with open(os.path.join(out_path, 'subject_ids.tsv'), 'w', newline='') as tsvfile_subs:
            writer_subs = csv.writer(tsvfile_subs, delimiter='\t', lineterminator='\n')
            writer_subs.writerow(["subject_id"])


            for grp in tqdm(sub_grp):
                sub_ids = getSubFolders(os.path.join(data_path,grp))

                writer_grps.writerow([grp] + sub_ids)
                writer_subs.writerows([[x] for x in sub_ids])

                for sub in sub_ids:
                    sessions = getSubFolders(os.path.join(data_path,grp,sub))

                    for sess in sessions:
                        monts = getSubFolders(os.path.join(data_path,grp,sub,sess))
                        
                        for mont in monts:
                            recs = getSubEDFFiles(os.path.join(data_path,grp,sub,sess, mont))
                            writer_recs.writerows([[grp, sub, sess, mont, rec, os.path.join(data_path,grp,sub,sess, mont, rec)] for rec in recs])


# with h5py.File(os.path.join(config.save_dir, 'predictions', name, rec[0] + '_' + rec[1] + '_preds.h5'), 'w') as f:
#                 f.create_dataset('y_pred', data=y_pred)
#                 f.create_dataset('y_true', data=y_true)

#             gc.collect()