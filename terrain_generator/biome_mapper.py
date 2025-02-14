def assign_biome(height, moisture):
    #mnich fahem bien , darha chatgpt apr n3wdha à 0
    """
    Assigns a biome based on height and moisture values.
    """
    if height < 0.2:
        return "ocean"
    elif height < 0.3:
        return "beach"
    elif height < 0.6:
        if moisture > 0.5:
            return "forest"
        else:
            return "grassland"
    else:
        return "mountain"