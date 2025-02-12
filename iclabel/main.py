# * Highpass filtering (0.25 to 0.75 Hz)
# * Reject channel if:
#     - flat > 20 secs
#     - SNR < .25 std devs 
#     - Pearson correlation < .6 Pearson of other channels
# * Decompose signals with Second Order Blind Identification ICA algorithm
# * Cluster ICs based on IClabel probs
# * If IC prob for (muscle, eye, heart, line noise, channel noise) > .6, then disregard

from rejecter import Rejecter

rejecter = Rejecter()
newchannels, kept = rejecter(channels=None)
