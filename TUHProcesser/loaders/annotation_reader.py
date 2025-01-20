import os
from typing import List, Tuple
import pandas as pd
import h5py


class Annotation_Reader:
    """ Class to store seizure annotations as read in the csv annotation files from the TUH dataset. """
    def __init__(
        self,
        events: List[Tuple[int, int]],
        annotation_path: str
    ):
        """Initiate an annotation instance

        Args:
            events (List([float,float])): list of tuples where each element contains the start and stop times in seconds of the event
            
        Returns:
            Annotation: returns an Annotation instance containing the events of the recording.
        """
        self.events = events
        self.annotation_path = annotation_path

    def getEvents(self):
        return self.events

    @classmethod
    def loadAnnotation(
        cls,
        annotation_path: str,
    ):
        szEvents = list()

        # If the recording path is given, convert it to the annotation path
        if annotation_path[-3:] == "edf":
            annotation_path = annotation_path[:-4] + ".csv_bi"
        try:
            csvFile = os.path.join(annotation_path)
            df = pd.read_csv(csvFile, header=5)
            for i, e in df.iterrows():
                if e['label'] != 'bckg':
                    szEvents.append([e['start_time'], e['stop_time']])
        except Exception as exc:
            print(exc)

        return cls(
            szEvents,
            annotation_path
        )

    def save_hdf5(self):
        """ Save the annotation as an HDF5 file """
        with h5py.File(self.rec_path[:-7]+'_annotation.h5', 'w') as f:
            f.create_dataset('events', data=self.events)