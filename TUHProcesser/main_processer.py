import os
from pathlib import Path
from TUHProcesser.classes.recording_reader import Recording_Reader
import csv

# Path where list of recordings is stored
out_path = Path('dataset_lists')

with open(os.path.join(out_path, 'recordings.tsv'), 'r', newline='') as tsvfile:
    reader = csv.reader(tsvfile, delimiter='\t', lineterminator='\n')
    next(reader)
    
    # For each EDF recording
    for row in reader:
        path = row[-1]

        # Remove non-EEG channels and standardize the channel names
        data = Recording_Reader.loadData(path)

        # Preprocess the recording
        data.preprocessData()

        # Channel configurations and sampling frequency can vary !!
        print('Sampling Frequency:',data.getFs())
        print('Montage:',data.getMontage())
        
        break