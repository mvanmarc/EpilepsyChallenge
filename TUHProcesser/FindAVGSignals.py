from pathlib import Path
from loaders.recording_reader import Recording_Reader
import numpy as np
from tqdm import tqdm
import itertools

rec_path_LE = Path("/esat/biomeddata/kkontras/TUH/tuh_eeg/tuh_eeg_seizure/v2.0.3/edf/dev/aaaaaayf/s001_2003/02_tcp_le/aaaaaayf_s001_t000.edf").as_posix()
rec_path_AVG = Path("/esat/biomeddata/kkontras/TUH/tuh_eeg/tuh_eeg_seizure/v2.0.3/edf/train/aaaaacle/s007_2010/03_tcp_ar_a/aaaaacle_s007_t000.edf").as_posix()
print("Loading File")
rec_reader = Recording_Reader.loadData(rec_path_LE)
ch_names = rec_reader.getChannelNames()
print(ch_names)
idxs = [x for x in range(len(ch_names))]
data = rec_reader.getData()
data = np.asarray(data)
print("Starting analyzing the subsets")
minimal = np.inf
minimalSubset = None
minimalSub_names = None
for l in tqdm(reversed(range(2,len(ch_names)+1))):
    for subset in tqdm(list(itertools.combinations(idxs, l))):
        res = data[subset,:]
        res = np.sum(res, axis=0)
        score = np.sum(np.abs(res)) 
        sub_names = [ch_names[x] for x in subset]
        if score < 100:
            print("Good subset found:")
            print(subset)
            print(sub_names)
            print(score)
            with open('goodSubsets.txt', 'a') as f:
                f.write(str(score) + '\n')
                f.write(str(subset) + '\n')
                f.write(str(sub_names) + '\n')
        if score < minimal:
            minimal = score
            minimalSubset = subset
            minimalSub_names = sub_names

    print("Minimal subset found:")
    print(minimalSubset)
    print(minimalSub_names)
    print(minimal)
    with open('minimalSubsets.txt', 'a') as f:
        f.write(str(minimal) + '\n')
        f.write(str(minimalSubset) + '\n')
        f.write(str(minimalSub_names) + '\n')
    minimal = np.inf
    minimalSubset = None
    minimalSub_names = None

