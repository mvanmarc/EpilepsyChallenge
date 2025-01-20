import copy
import os
from pathlib import Path
from loaders.recording_reader import Recording_Reader
from loaders.annotation_reader import Annotation_Reader
from loaders.checker import Checker
import csv
import argparse
from tqdm import tqdm
from joblib import Parallel, delayed


def process_one_subject(path: str, fs_resamp = 200,
                butter_order = 4, fc_low = 0.5, fc_high = 100)-> bool:
    """ Process one subject's EDF recordings. """

    checker = Checker()

    try:
        data = Recording_Reader.loadData(path)
        annotation = Annotation_Reader.loadAnnotation(path)
        if not checker(data, annotation):
            raise Exception(str(checker))
        data.preprocessData(fs_resamp, butter_order, fc_low , fc_high)
        data.save_hdf5(annotation)

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
    parser.add_argument('--fs_resamp', default = 200, action='store_true', help='Resampling Rate')
    parser.add_argument('--butter_order', default = 4, action='store_true', help = 'Order of the Butterworth filters for preprocessing')
    parser.add_argument('--fc_low', default = 0.5, action='store_true', help='lower cut-off frequency for bandpass')
    parser.add_argument('--fc_high', default = 100, action='store_true', help='higher cut-off frequency for bandpass')
    args = parser.parse_args()

    with open(Path(args.rec_path), 'r', newline='') as tsvfile:
        reader = csv.reader(tsvfile, delimiter='\t', lineterminator='\n')
        next(reader) # Skip the header
        reader = list(reader)

        if args.parallel:
            #catch boolen values to see if all were successful
            success = Parallel(n_jobs=-1)(delayed(process_one_subject)(row[-1], args.fs_resamp, args.butter_order, args.fc_low, args.fc_high) for i, row in tqdm(enumerate(reader), total=len(reader))) #run the first 100 recordings
            print("Successful cases: ", success.count(True))
            print("Failed cases: ", success.count(False))
            # Parallel(n_jobs=-1)(delayed(process_one_subject)(row[-1]) for row in tqdm(reader))
        else:
            for i, row in tqdm(enumerate(reader)):
                process_one_subject(row[-1])
                # if i == 10: break
