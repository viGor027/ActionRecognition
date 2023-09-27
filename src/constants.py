import os

ROOT_PATH = "../data"

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
