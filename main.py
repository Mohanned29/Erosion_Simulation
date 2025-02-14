from terrain_generator.noise_generator import generate_heightmap
from terrain_generator.biome_mapper import assign_biome
from terrain_generator.erosion_simulator import thermal_erosion
from terrain_generator.terrain_renderer import TerrainRenderer
from terrain_generator.utils import normalize_heightmap

width, height = 100, 100
heightmap = generate_heightmap(width, height)
heightmap = thermal_erosion(heightmap)
heightmap = normalize_heightmap(heightmap)

renderer = TerrainRenderer(heightmap)
renderer.run()