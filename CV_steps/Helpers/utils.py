import numpy as np

def normalize(x):
    x = np.asarray(x)
    return (x - x.min()) / (np.ptp(x))