import copy
import os
from pathlib import Path
from loaders.recording_reader import Recording_Reader
from loaders.annotation_reader import Annotation_Reader
import csv
import argparse
from tqdm import tqdm
from joblib import Parallel, delayed


def process_one_subject(path: str)-> bool:
    """ Process one subject's EDF recordings. """

    try:
        data = Recording_Reader.loadData(path)
        data.preprocessData()
        data.save_hdf5()
        annotation = Annotation_Reader.loadAnnotation(path)
        annotation.save_hdf5()

    except Exception as e:
        #save all such errors in a common file
        with open('error_file.txt', 'a') as f:
            f.write(path + '\n')
            f.write(str(e) + '\n')
        return False
    return True # Success

#main
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Heatmap of Tuning')
    parser.add_argument('--rec_path', default="./dataset_lists/recordings.tsv", type=str, help='Path to the recordings.tsv file')
    parser.add_argument('--parallel', action='store_true', help='Parallel processing')
    args = parser.parse_args()

    with open(Path(args.rec_path), 'r', newline='') as tsvfile:
        reader = csv.reader(tsvfile, delimiter='\t', lineterminator='\n')
        next(reader) # Skip the header
        reader = list(reader)

        if args.parallel:
            #catch boolen values to see if all were successful
            success = Parallel(n_jobs=-1)(delayed(process_one_subject)(row[-1]) for i, row in tqdm(enumerate(reader), total=len(reader))) #run the first 100 recordings
            print("Successful cases: ", success.count(True))
            print("Failed cases: ", success.count(False))
            # Parallel(n_jobs=-1)(delayed(process_one_subject)(row[-1]) for row in tqdm(reader))
        else:
            for i, row in tqdm(enumerate(reader)):
                process_one_subject(row[-1])
                if i == 10: break
