"""
Data Preprocessing and Generator Configuration

This script defines functions and configurations related to data preprocessing
and generator creation for training, validation, and testing.

Module Dependencies:
    - cv2 (imported as cv2)
    - os
    - numpy as np
    - random
    - tf (imported as tensorflow) with dependencies
        - src.constants (imported as CLASSES, ClASS_PATHS, STARTING_FRAMES,
                            TOTAL_N_TRAINING_SAMPLES, IMAGE_SIZE, N_FRAMES)

Global Constants:
    - batch_size (int): Batch size for training, validation, and testing datasets.
    - x_shape (tuple): Shape of input data (number of frames, height, width, channels).
    - y_shape (tuple): Shape of output labels (number of classes).
    - x_type (tf.dtype): Data type of input.
    - y_type (tf.dtype): Data type of output.

Global Dataset Objects:
    - train_ds (tf.data.Dataset): Training dataset object.
    - val_ds (tf.data.Dataset): Validation dataset object.
    - test_ds (tf.data.Dataset): Testing dataset object.

Example Usage:
    - Utilize this script to define functions for preprocessing video data and creating data generators.
    - Adjust batch_size, x_shape, and y_shape based on the model and task requirements.
    - Configure dataset objects train_ds, val_ds, and test_ds for training, validation, and testing datasets.
"""

import cv2
import os
import numpy as np
import random
from src.constants import CLASSES, ClASS_PATHS, STARTING_FRAMES,\
    TOTAL_N_TRAINING_SAMPLES, IMAGE_SIZE, N_FRAMES
import tensorflow as tf


def preprocess_video_file(filepath: str, n_frames: int = 5) -> np.ndarray:
    """
    Preprocesses a video file by extracting frames.

    Parameters:
        filepath (str): Path to the video file.
        n_frames (int): Number of frames to extract.

    Returns:
        np.ndarray: Processed frames as a NumPy array.
    """
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
        frame = cv2.resize(frame, (IMAGE_SIZE, IMAGE_SIZE))

        if ret:
            frames.append(frame)
            frame_count += 1
            current_frame += step
        else:
            break

    cap.release()

    return np.array(frames)


def get_set_of_videos(set_type: str, list_of_videos: list) -> list:
    """
    Gets a subset of videos based on the set type (train, valid, test).

    Parameters:
        set_type (str): Type of the dataset (train, valid, test).
        list_of_videos (list): List of video filenames.

    Returns:
        list: Subset of videos.
    """
    n_videos = len(list_of_videos)
    if set_type == 'train':
        return list_of_videos[: int(.8 * n_videos)]
    elif set_type == 'valid':
        return list_of_videos[int(.8 * n_videos): int(.9 * n_videos)]
    else:
        return list_of_videos[int(.9 * n_videos):]


def get_gen(set_type: str) -> callable:
    """
    Creates a generator function for generating batches of preprocessed video data and labels.

    Parameters:
        set_type (str): Type of the dataset (train, valid, test).

    Returns:
        callable: Generator function.
    """
    def gen():
        """
        Generator function for creating batches of preprocessed video data and labels.

        Yields:
            tuple: A tuple containing preprocessed video data and corresponding labels.

        Notes:
            - This generator function is designed to be used with TensorFlow Dataset.from_generator().

        Example:
            Usage with tf.data.Dataset.from_generator():

            ```python
            train_ds = tf.data.Dataset.from_generator(get_gen('train'), output_signature=(
                tf.TensorSpec(shape=x_shape, dtype=x_type),
                tf.TensorSpec(shape=y_shape, dtype=y_type),))
            ```
        """
        for category, cls in enumerate(CLASSES):
            videos = os.listdir(ClASS_PATHS[cls])
            videos_for_set_type = get_set_of_videos(set_type, videos)
            for video_path in videos_for_set_type:
                if set_type == 'train' and (cls == 0 or cls == 1) and np.random.randint(1, 101) <= 30:
                    yield preprocess_video_file(
                        os.path.join(ClASS_PATHS[cls], video_path), n_frames=N_FRAMES), (category,)
                yield preprocess_video_file(
                    os.path.join(ClASS_PATHS[cls], video_path), n_frames=N_FRAMES), (category,)

    return gen


batch_size = 64

x_shape = (N_FRAMES, IMAGE_SIZE, IMAGE_SIZE, 3)
y_shape = (1,)

x_type = tf.float32
y_type = tf.int8

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
