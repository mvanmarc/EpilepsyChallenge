from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

def getSubFolders(path: Path) -> list:
    """ Get list of all subfolders in a given folder """
    try:
        res = [name for name in os.listdir(path) if os.path.isdir(os.path.join(path,name))]
        return sorted(res)
    except Exception as exc:
        print(exc)
        return []

def getSubFiles(path: Path, type:str) -> list:
    """ Get list of all EDF files (.edf) in a given folder """
    try:
        res = [name for name in os.listdir(path) if os.path.isfile(os.path.join(path,name)) and name[-len(type):] == type]
        return sorted(res)
    except Exception as exc:
        print(exc)
        return []
    
def isAnnotationAvailable(path: Path) -> bool:
    """ Check whether a given EDF file has an annotation file available. """
    ann_path = path[:-4] + ".csv_bi"
    return os.path.isfile(ann_path)

def pre_process(ch_data, fs_data, fs_resamp = 200, 
                butter_order = 4, fc_low = 0.5, fc_high = 100):
    """ Resample and filter given channel """
    
    ## Filtering : 0.5 - 100 Hz

    b, a = signal.butter(butter_order, fc_low/(fs_data/2), 'high')
    ch_data = signal.filtfilt(b, a, ch_data)

    b, a = signal.butter(butter_order, fc_high/(fs_data/2), 'low')
    ch_data = signal.filtfilt(b, a, ch_data)

    ## Powerline-interference filtering : 50 and 60 Hz

    b, a = signal.butter(butter_order, [59.5/(fs_data/2), 60.5/(fs_data/2)], 'bandstop')
    ch_data = signal.filtfilt(b, a, ch_data)
    
    b, a = signal.butter(butter_order, [49.5/(fs_data/2), 50.5/(fs_data/2)], 'bandstop')
    ch_data = signal.filtfilt(b, a, ch_data)


    ## Resampling to 200 Hz

    if fs_resamp != fs_data:
        ch_data = signal.resample(ch_data, int(fs_resamp*len(ch_data)/fs_data))

    return ch_data, fs_resamp

def make_hdf5(ch_data):
    pass

def mask2eventList(mask, fs):
    """ Convert array of 0/1's to list of events in seconds. """

    events = list()
    tmp = []
    start_i = np.where(np.diff(np.array(mask, dtype=int)) == 1)[0]
    end_i = np.where(np.diff(np.array(mask, dtype=int)) == -1)[0]
    
    if len(start_i) == 0 and mask[0]:
        events.append([0, (len(mask))/fs])  # DIFFERENT FROM MIGUEL!!! :  events.append([0, (len(mask)-1)/fs]) 
    else:
        # Edge effect
        if mask[0]:
            events.append([0, (end_i[0]+1)/fs])
            end_i = np.delete(end_i, 0)
        # Edge effect
        if mask[-1]:
            if len(start_i):
                tmp = [[(start_i[-1]+1)/fs, (len(mask))/fs]]
                start_i = np.delete(start_i, len(start_i)-1)
        for i in range(len(start_i)):
            events.append([(start_i[i]+1)/fs, (end_i[i]+1)/fs])
        events += tmp
    return events

def standardizeEEGChannelName(name):
    """ Standardize the EEG channel names from "EEG FP1-REF" to "Fp1" """
    return name.lower().split(" ")[1].split("-")[0]
    
def extractMontage(name):
    """ Extract the montage (given a channel name) """
    return name.split(" ")[1].split("-")[1]