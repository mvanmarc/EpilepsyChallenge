from loaders.annotation_reader import Annotation_Reader
from loaders.recording_reader import Recording_Reader
import numpy as np

class Checker:

    def __init__(self):
        self.lastMsg = ""
        self.minimumChannels = ["fp1", "f3", "c3", "p3", "o1", "f7", "t3", "t5", "fz", "cz", "pz", "fp2", "f4", "c4", "p4", "o2", "f8", "t4", "t6"]
        pass

    def __str__(self):
        return self.lastMsg

    def _allChannelsPresent(self, recording: Recording_Reader, annotation: Annotation_Reader):
        recordingChannels = recording.getChannelNames()
        for ch in self.minimumChannels:
            if ch not in recordingChannels:
                self.lastMsg = "Missing Channel: Channel "+ch+" not in "+str(recordingChannels)
                return False
        return True

    def _isAnnotationValid(self, recording: Recording_Reader, annotation: Annotation_Reader):
        duration = annotation.getDuration()
        for event in annotation.getEvents():
            if 0 > event[0] or event[0]>event[1] or event[1]>duration:
                self.lastMsg = "Invalid annotation: "+str(event)
                return False
        return True

    def _channelsNotShort(self, recording: Recording_Reader, annotation: Annotation_Reader):
        channels = recording.getData()
        fs = recording.getFs()
        for i in range(len(channels)):
            if len(channels[i]) < fs:
                self.lastMsg = "Recording far too short (smaller than one second): "+str(len(channels[i])/fs) + " second."
                return False
        return True
    
    def _channelsNotEmpty(self, recording: Recording_Reader, annotation: Annotation_Reader):
        channels = recording.getChannelNames()
        for i in range(len(channels)):
            if channels[i] in self.minimumChannels:
                ch = np.array(recording.getChannel(i))
                if np.sum(np.diff(ch) == 0)>len(ch)/2:
                    self.lastMsg = "More than half of the recording stays the same value: "+ str(np.sum(np.diff(ch) == 0)/len(ch))
                    return False
        return True
    
    # def _isReferenceMontage(self, recording: Recording_Reader, annotation: Annotation_Reader):
    #     if recording.getMontage().lower() != "ref" and recording.getMontage().lower() != "avg" :
    #                 self.lastMsg = "Not in a referential montage: "+ recording.getMontage()
    #                 return False
    #     return True
    
    def __call__(self, recording: Recording_Reader, annotation: Annotation_Reader):
        checkFunctions = [self._isAnnotationValid,
                          self._allChannelsPresent,
                          self._channelsNotShort,
                          self._channelsNotEmpty]
        return all([func(recording, annotation) for func in checkFunctions])