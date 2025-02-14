import numpy as np
import noise

#generation of heightmap using perlin noise stuff (new to this)
def generate_heightmap(width, height, scale=50.0, octaves=6, persistence=0.5, lacunarity=2.0):

    world = np.zeros((width, height))
    for i in range(width):
        for j in range(height):
            world[i][j] = noise.pnoise2(i/scale, j/scale,octaves=octaves,persistence=persistence,lacunarity=lacunarity)
    return world