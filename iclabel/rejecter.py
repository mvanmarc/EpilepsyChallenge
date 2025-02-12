import numpy as np
from scipy.stats import pearsonr

class Rejecter:
    # * Reject channel if:
    # - flat > 20 secs
    # - SNR < .25 std devs 
    # - Pearson correlation < .6 Pearson of other channels
    def __init__(self, snrThreshold = .25, pearsonThreshold = .6):
        self.snrThreshold = snrThreshold
        self.pearsonThreshold = pearsonThreshold
        pass

    def __call__(self, channels):
        res = self.isFlat(channels)
        res = np.logical_and(res, self.hasLowSNR(channels))
        res = np.logical_and(res, self.hasLowPearson(channels))
        return channels[res,:], res

    def isFlat(self, channels):
        return np.ones(np.size(channels,0))

    def hasLowSNR(self, channels):
        raise Exception()
        return 
    
    def hasLowPearson(self, channels):
        res = np.ones(np.size(channels,0))
        flag = True
        while flag == True and np.sum(res)>1:
            flag = False
            for i in range(len(res)):
                if res[i] == 1:
                    if pearsonr(channels[i,:], np.mean(np.delete(channels[res,:], i))) < self.pearsonThreshold :
                        res[i] = 0
                        flag = True
        return res
    

