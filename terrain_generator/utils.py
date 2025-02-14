import numpy as np
def normalize_heightmap(heightmap):
    """
    Normalizes heightmap values to [0, 1].
    """
    return (heightmap - np.min(heightmap)) / (np.max(heightmap) - np.min(heightmap))