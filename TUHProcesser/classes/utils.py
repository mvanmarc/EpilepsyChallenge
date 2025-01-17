from scipy import signal
import numpy as np
import matplotlib.pyplot as plt

def pre_process(ch_data, fs_data, american = False):
    """ Resample and filter given channel """

    ## Resampling to 200 Hz

    fs_resamp = 200

    if fs_resamp != fs_data:
        ch_data = signal.resample(ch_data, int(fs_resamp*len(ch_data)/fs_data))
    
    ## Filtering : 0.5 - 60 Hz

    b, a = signal.butter(4, 0.5/(fs_resamp/2), 'high')
    ch_data = signal.filtfilt(b, a, ch_data)

    b, a = signal.butter(4, 60/(fs_resamp/2), 'low')
    ch_data = signal.filtfilt(b, a, ch_data)

    ## Powerline-interference filtering : 50 or 60 Hz

    if american: # 60 Hz Notch
        b, a = signal.butter(4, [59.5/(fs_resamp/2), 60.5/(fs_resamp/2)], 'bandstop')
        ch_data = signal.filtfilt(b, a, ch_data)
    else: # 50 Hz Notch
        b, a = signal.butter(4, [49.5/(fs_resamp/2), 50.5/(fs_resamp/2)], 'bandstop')
        ch_data = signal.filtfilt(b, a, ch_data)

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
    """ Standardize the EEG channel names from "EEG F1-REF" to "F1" """
    return name.split(" ")[1].split("-")[0]

def extractMontage(name):
    """ Extract the montage (given a channel name) """
    return name.split(" ")[1].split("-")[1]