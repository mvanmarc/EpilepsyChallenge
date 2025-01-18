import os
from pathlib import Path
from TUHProcesser.loaders.recording_reader import Recording_Reader
import csv
import argparse
from tqdm import tqdm
from joblib import Parallel, delayed


def process_one_subject(path: str)-> bool:
    """ Process one subject's EDF recordings. """
    data = Recording_Reader.loadData(path)

    # Preprocess the recording
    data.preprocessData()

    data.save_hdf5()

    # Channel configurations and sampling frequency can vary !!
    # print('Sampling Frequency:', data.getFs())
    # print('Montage:', data.getMontage())
    return True # Success

#main
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Heatmap of Tuning')
    parser.add_argument('--rec_path', default="dataset_lists/recordings.tsv", type=str, help='Path to the recordings.tsv file')
    parser.add_argument('--parallel', action='store_true', help='Parallel processing')
    args = parser.parse_args()

    with open(Path(args.rec_path), 'r', newline='') as tsvfile:
        reader = csv.reader(tsvfile, delimiter='\t', lineterminator='\n')
        next(reader)
        print(args.parallel)
        if args.parallel:
            Parallel(n_jobs=-1)(delayed(process_one_subject)(row[-1]) for row in tqdm(reader))
        else:
            for id, row in tqdm(enumerate(reader)):
                path = row[-1]
                process_one_subject(path)
                if id == 10:
                    break