import cv2
import os
import numpy as np
import random
from constants import CLASSES, ClASS_PATHS, STARTING_FRAMES, TOTAL_N_TRAINING_SAMPLES
import tensorflow as tf


def preprocess_video_file(filepath: str, n_frames: int = 5):
    cap = cv2.VideoCapture(filepath)
    frames = []
    frame_count = 0

    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    starting_frame = int(random.choice(STARTING_FRAMES) * total_frames)
    if total_frames < starting_frame + n_frames:
        raise ValueError("starting_frame needs to be lower than total number of frames in the video"
                         "minus the number of frames you want to extract")

    current_frame = starting_frame
    step = (total_frames - starting_frame) // n_frames
    while cap.isOpened() and frame_count < n_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        ret, frame = cap.read()
        frame = cv2.resize(frame, (128, 128))

        if ret:
            frames.append(frame)
            frame_count += 1
            current_frame += step
        else:
            break

    cap.release()

    return np.array(frames)


def get_set_of_videos(set_type: str, list_of_videos: list):
    n_videos = len(list_of_videos)
    if set_type == 'train':
        return list_of_videos[: int(.8 * n_videos)]
    elif set_type == 'valid':
        return list_of_videos[int(.8 * n_videos): int(.9 * n_videos)]
    else:
        return list_of_videos[int(.9 * n_videos):]


def get_gen(set_type: str):
    def gen():
        for category, cls in enumerate(CLASSES):
            videos = os.listdir(ClASS_PATHS[cls])
            videos_for_set_type = get_set_of_videos(set_type, videos)
            for video_path in videos_for_set_type:
                yield preprocess_video_file(
                    os.path.join(ClASS_PATHS[cls], video_path)), (category,)

    return gen


batch_size = 64

x_shape = (5, 128, 128, 3)
y_shape = (1,)

x_type = tf.float32
y_type = tf.float32

train_ds = tf.data.Dataset.from_generator(get_gen('train'), output_signature=(
    tf.TensorSpec(shape=x_shape, dtype=x_type),
    tf.TensorSpec(shape=y_shape, dtype=y_type),))

train_ds = train_ds.shuffle(TOTAL_N_TRAINING_SAMPLES // 2)
train_ds = train_ds.batch(batch_size)
train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

val_ds = tf.data.Dataset.from_generator(get_gen('valid'), output_signature=(
    tf.TensorSpec(shape=x_shape, dtype=x_type),
    tf.TensorSpec(shape=y_shape, dtype=y_type),))

val_ds = val_ds.batch(batch_size)
val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

test_ds = tf.data.Dataset.from_generator(get_gen('test'), output_signature=(
    tf.TensorSpec(shape=x_shape, dtype=x_type),
    tf.TensorSpec(shape=y_shape, dtype=y_type),))

test_ds = test_ds.batch(batch_size)
test_ds = test_ds.prefetch(tf.data.AUTOTUNE)
