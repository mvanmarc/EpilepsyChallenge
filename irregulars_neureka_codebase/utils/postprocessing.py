import numpy as np
from scipy import signal


def apply_preprocess_eeg(rec):
    idx_focal = [i for i, c in enumerate(rec.channels) if c == 'BTEleft SD']
    if not idx_focal:
        idx_focal = [i for i, c in enumerate(rec.channels) if c == 'BTEright SD']
    idx_cross = [i for i, c in enumerate(rec.channels) if c == 'CROSStop SD']
    if not idx_cross:
        idx_cross = [i for i, c in enumerate(rec.channels) if c == 'BTEright SD']

    ch_focal, _ = pre_process_ch(rec.data[idx_focal[0]], rec.fs[idx_focal[0]], 250)
    ch_cross, _ = pre_process_ch(rec.data[idx_cross[0]], rec.fs[idx_cross[0]], 250)

    return [ch_focal, ch_cross]


def pre_process_ch(ch_data, fs_data, fs_resamp):
    if fs_resamp != fs_data:
        ch_data = signal.resample(ch_data, int(fs_resamp * len(ch_data) / fs_data))

    b, a = signal.butter(4, 0.5 / (fs_resamp / 2), 'high')
    ch_data = signal.filtfilt(b, a, ch_data)

    b, a = signal.butter(4, 60 / (fs_resamp / 2), 'low')
    ch_data = signal.filtfilt(b, a, ch_data)

    b, a = signal.butter(4, [49.5 / (fs_resamp / 2), 50.5 / (fs_resamp / 2)], 'bandstop')
    ch_data = signal.filtfilt(b, a, ch_data)

    return ch_data, fs_resamp


def eventList2Mask(events, totalLen, fs):
    """Convert list of events to mask.

    Returns a logical array of length totalLen.
    All event epochs are set to True

    Args:
        events: list of events times in seconds. Each row contains two
                columns: [start time, end time]
        totalLen: length of array to return in samples
        fs: sampling frequency of the data in Hertz
    Return:
        mask: logical array set to True during event epochs and False the rest
              if the time.
    """
    mask = np.zeros((totalLen,))
    for event in events:
        for i in range(min(int(event[0] * fs), totalLen), min(int(event[1] * fs), totalLen)):
            mask[i] = 1
    return mask


def mask2eventList(mask, fs):
    """Convert mask to list of events.

    Args:
        mask: logical array set to True during event epochs and False the rest
          if the time.
        fs: sampling frequency of the data in Hertz
    Return:
        events: list of events times in seconds. Each row contains two
                columns: [start time, end time]
    """
    events = list()
    tmp = []
    start_i = np.where(np.diff(np.array(mask, dtype=int)) == 1)[0]
    end_i = np.where(np.diff(np.array(mask, dtype=int)) == -1)[0]

    if len(start_i) == 0 and mask[0]:
        events.append([0, (len(mask) - 1) / fs])
    else:
        # Edge effect
        if mask[0]:
            events.append([0, (end_i[0] + 1) / fs])
            end_i = np.delete(end_i, 0)
        # Edge effect
        if mask[-1]:
            if len(start_i):
                tmp = [[(start_i[-1] + 1) / fs, (len(mask)) / fs]]
                start_i = np.delete(start_i, len(start_i) - 1)
        for i in range(len(start_i)):
            events.append([(start_i[i] + 1) / fs, (end_i[i] + 1) / fs])
        events += tmp
    return events


def merge_events(events, distance):
    """ Merge events.

    Args:
        events: list of events times in seconds. Each row contains two
                columns: [start time, end time]
        distance: maximum distance (in seconds) between events to be merged
    Return:
        events: list of events (after merging) times in seconds.
    """
    i = 1
    tot_len = len(events)
    while i < tot_len:
        if events[i][0] - events[i - 1][1] < distance:
            events[i - 1][1] = events[i][1]
            events.pop(i)
            tot_len -= 1
        else:
            i += 1
    return events

def skip_events(events, margin):
    ev_list = []
    for i in range(len(events)):
        if events[i][1] - events[i][0] >= margin * 0.8:
            ev_list.append(events[i])
    return ev_list

def post_events(events, margin):
    ''' Converts the unprocessed events to the post-processed events based on physiological constrains:
    - seizure alarm events distanced by 0.2*margin (in seconds) are merged together
    - only events with a duration longer than margin*0.8 are kept
    (for more info, check: K. Vandecasteele et al., “Visual seizure annotation and automated seizure detection using
    behind-the-ear elec- troencephalographic channels,” Epilepsia, vol. 61, no. 4, pp. 766–775, 2020.)

    Args:
        events: list of events times in seconds. Each row contains two
                columns: [start time, end time]
        margin: float, the desired margin in seconds

    Returns:
        ev_list: list of events times in seconds after merging and discarding short events.
    '''
    events_merge = merge_events(events, 0.2*margin)

    skipped_events = skip_events(events_merge, margin)

    return skipped_events


def post_processing(y_pred, fs, th, margin):
    ''' Post process the predictions given by the model based on physiological constraints: a seizure is
    not shorter than 10 seconds and events separated by 2 seconds are merged together.

    Args:
        y_pred: array with the seizure classification probabilties (of each segment)
        fs: sampling frequency of the y_pred array (1/window length - in this challenge fs = 1/2)
        th: threshold value for seizure probability (float between 0 and 1)
        margin: float, the desired margin in seconds (check get_events)

    Returns:
        pred: array with the processed classified labels by the model
    '''
    pred = (y_pred > th)
    events = mask2eventList(pred, fs)
    events = post_events(events, margin)

    return events

