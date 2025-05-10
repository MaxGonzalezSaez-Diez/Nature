# Utils file with some helper functions for the main script

def get_spe_nr(idx):
    """Get the species number from the agent ID"""
    if idx == "p":
        return 0
    return int(idx.split("_")[1].replace("s", ""))


def strip_idNr(idx):
    """Get the level and species number from the agent ID"""
    return "_".join(idx.split("_")[:2])


def get_level_nr(idx):
    """Get the level number from the agent ID"""
    if idx == "p":
        return 0
    return int(idx.split("_")[0].replace("l", ""))
