import numpy as np

def thermal_erosion(heightmap, iterations=10, erosion_rate=0.01):

    for _ in range(iterations):
        for i in range(1, heightmap.shape[0]-1):
            for j in range(1, heightmap.shape[1]-1):
                neighbors = [
                    heightmap[i-1][j], heightmap[i+1][j],
                    heightmap[i][j-1], heightmap[i][j+1]
                ]
                max_diff = max(neighbors) - heightmap[i][j]
                if max_diff > 0:
                    heightmap[i][j] -= erosion_rate * max_diff
                    heightmap[np.argmax(neighbors)] += erosion_rate * max_diff
    return heightmap