import os
from pathlib import Path
import random
import numpy as np
import shutil


BASE_PATH = os.path.dirname(os.path.realpath(__file__))
DATA_PATH = Path(BASE_PATH + '/data')
TRAIN_PATH = Path(BASE_PATH + '/train')
VALID_PATH = Path(BASE_PATH + '/valid')


def prepare_files():
    if os.path.isdir(str(TRAIN_PATH)) and os.path.isdir(str(VALID_PATH)):
        return

    image_count = get_dataset_size()
    cross_validation_indexes = get_cross_validation_indexes(image_count)
    
    all_image_paths = list(DATA_PATH.glob('*/*'))
    
    for i in range(image_count):
        if i in cross_validation_indexes:
            move_image_to_validation(str(all_image_paths[i]))
        else:
            move_image_to_training(str(all_image_paths[i]))


def get_nb_training_files():
    return sum([len(files) for r, d, files in os.walk(str(TRAIN_PATH))])


def get_nb_validation_files():
    return sum([len(files) for r, d, files in os.walk(str(VALID_PATH))])


def move_image_to_training(image_path):
    label = os.path.basename(os.path.dirname(image_path))
    new_location = str(TRAIN_PATH) + '/' + label
    if not os.path.exists(new_location):
        os.makedirs(new_location)
    shutil.move(image_path, new_location)


def move_image_to_validation(image_path):
    label = os.path.basename(os.path.dirname(image_path))
    new_location = str(VALID_PATH) + '/' + label
    if not os.path.exists(new_location):
        os.makedirs(new_location)
    shutil.move(image_path, str(new_location))


def get_dataset_size():
    return sum([len(files) for _, _, files in os.walk(str(DATA_PATH))])


def get_cross_validation_indexes(dataset_size, validation_set_percentage = 0.2):
    number_of_values = int(dataset_size * validation_set_percentage)
    indexes = np.random.permutation(dataset_size)

    return indexes[0:number_of_values]