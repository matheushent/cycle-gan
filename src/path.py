import os

join = os.path.join

def mkdir(path):
    """
    Utility function to create directories

    Args:
        path (str): Path to be created
    """

    if not os.exists(path):
        os.makedirs(path)