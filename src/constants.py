"""
Data Configuration and Constants

This script defines constants and configurations related to the data used for training and evaluation.

Module Dependencies:
    - os (imported as os)

Global Constants:
    - ROOT_PATH (str): Root directory path for the data.
    - CLASSES (list): List of class labels derived from subdirectories in the data root.
    - STARTING_FRAMES (list): List of starting frames ratios for sequence sampling.
    - CLASS_PATHS (dict): Dictionary mapping class labels to their respective data paths.
    - TOTAL_N_TRAINING_SAMPLES (int): Total number of training samples calculated as 80% of the dataset.
    - IMAGE_SIZE (int): Size of the input images (height, width) for the model.
    - N_FRAMES (int): Number of frames in each sequence for sequence-based models.

Example Usage:
    - Use this script to define data-related constants and configurations for training and evaluation.
    - Adjust the constants based on the specifics of the dataset and the task requirements.

Note:
    - Ensure that the data directory structure matches the expected structure based on the provided constants.

"""

import os

ROOT_PATH = os.path.join((os.path.dirname(os.path.dirname(__file__))), 'data')

CLASSES = os.listdir(ROOT_PATH)

for item_id, item in enumerate(CLASSES):
    print(item_id, item)

STARTING_FRAMES = [.05, .1, .15, .20]

ClASS_PATHS = {}

TOTAL_N_TRAINING_SAMPLES = 0

for cls in CLASSES:
    ClASS_PATHS[cls] = os.path.join(ROOT_PATH, cls)
    TOTAL_N_TRAINING_SAMPLES += len(os.listdir(ClASS_PATHS[cls]))

TOTAL_N_TRAINING_SAMPLES = int(TOTAL_N_TRAINING_SAMPLES * 0.8)

IMAGE_SIZE = 224
N_FRAMES = 6
