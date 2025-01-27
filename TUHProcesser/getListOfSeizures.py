from pathlib import Path
from loaders.recording_reader import Recording_Reader
from loaders.annotation_reader import Annotation_Reader
from loaders.checker import Checker
import csv
import argparse
from tqdm import tqdm
import os


def process_all_subjects(reader, fs_resamp = 200,
                butter_order = 4, fc_low = 0.5, fc_high = 100)-> bool:
    """ Process one subject's EDF recordings. """
    lst = []
    maxlen = 0
    bestpath = ""
    for i, row in tqdm(enumerate(reader), total=len(reader)):
        path = row[-1]
        # checker = Checker()

        # data = Recording_Reader.loadData(path)
        annotation = Annotation_Reader.loadAnnotation(path)
        # if not checker(data, annotation):
        #     continue
        evs = annotation.getEvents()
        if len(evs) > 0 and os.path.exists(path[:-4]+".h5"):
            lst.append([len(evs), path])
            if len(evs)>maxlen:
                maxlen = len(evs)
                bestpath = path
    return lst, maxlen, bestpath # Success



#main
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Heatmap of Tuning')
    parser.add_argument('--rec_path', default="./TUHProcesser/dataset_lists/recordings.tsv", type=str, help='Path to the recordings.tsv file')
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
        lst, maxlen, bestpath = process_all_subjects(reader)
        print("Most seizures in one recording:")
        print(maxlen)
        print(bestpath)
        with open(os.path.join(Path('./TUHProcesser/dataset_lists'), 'seizures.tsv'), 'w', newline='') as tsvfile_recs:
            writer_recs = csv.writer(tsvfile_recs, delimiter='\t', lineterminator='\n')
            writer_recs.writerow(["#Seizures", "Path"])
            writer_recs.writerows(lst)
                    
