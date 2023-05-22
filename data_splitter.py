"""Data splitter for Pre-training and Fine-tuning phases."""
import os
import glob
import numpy as np
import shutil
import splitfolders


_BASE_PATH = os.environ['_BASE_PATH']
_SPLIT_PATH = os.environ['_SPLIT_PATH']
NUM_PRETRAIN_CLASSES = int(os.environ['_NUM_PRETRAIN_CLASSES'])
NUM_FINETUNE_CLASSES = int(os.environ['_NUM_FINETUNE_CLASSES'])
PRETRAIN_FOLDER_NAME = os.environ['_PRETRAIN_FOLDER_NAME']
FINETUNE_FOLDER_NAME = os.environ['_FINETUNE_FOLDER_NAME']


def load_data():
    """Load and shuffle data from the base path."""

    # get all character folders
    character_folders = glob.glob(
        os.path.join(_BASE_PATH, '*/*/')
    )
    assert len(character_folders) == NUM_PRETRAIN_CLASSES + NUM_FINETUNE_CLASSES

    # shuffle characters
    np.random.default_rng(0).shuffle(character_folders)

    return character_folders


def split_data(character_folders):
    """Split data into pretrain and finetune sets."""

    # split data into train, val, and test sets
    pretrain_folders = character_folders[:NUM_PRETRAIN_CLASSES]
    finetune_folders = character_folders[NUM_PRETRAIN_CLASSES:]

    return pretrain_folders, finetune_folders


def make_dataset(character_folders, split):
    """Create dataset folder for train, val, and test sets."""

    os.makedirs(os.path.join(_SPLIT_PATH, split), exist_ok=True)

    for character_path in character_folders:
        writing_system, character = character_path.split('/')[-3:-1]

        src_path = character_path
        dst_path = os.path.join(_SPLIT_PATH, split, f"{writing_system}_{character}")
        shutil.copytree(src_path, dst_path)


def train_val_test_split(folder_name):
    """Split data at folder_name into train, val, and test sets."""

    input_directory = os.path.join(_SPLIT_PATH, folder_name)

    splitfolders.ratio(
        input_directory, # The location of dataset
        output=input_directory, # The output location
        seed=0, # The number of seed
        ratio=(.5, .2, .3), # The ratio of split dataset
        group_prefix=None, # If your dataset contains more than one file like ".jpg", ".pdf", etc
        move=True # If you choose to move, turn this into True
    )


def clean_up(folder_name):
    """Remove character folders after splitting data."""

    character_folders = glob.glob(
        os.path.join(_SPLIT_PATH, folder_name, '*/')
    )

    for character_path in character_folders:
        character_folder_name = character_path.split('/')[-2]
        if character_folder_name in ['train', 'val', 'test']:
            continue
        else:
            shutil.rmtree(character_path)


def main():
    # Make data dir
    os.makedirs(_SPLIT_PATH, exist_ok=True)

    # Load and shuffle data
    character_folders = load_data()

    # Split data into pretrain and finetune sets
    pretrain_folders, finetune_folders = split_data(character_folders)

    # Move data into pretrain and finetune sets
    make_dataset(pretrain_folders, PRETRAIN_FOLDER_NAME)
    make_dataset(finetune_folders, FINETUNE_FOLDER_NAME)

    # Split data into train, val, and test sets
    train_val_test_split(PRETRAIN_FOLDER_NAME)
    train_val_test_split(FINETUNE_FOLDER_NAME)

    # Clean up
    clean_up(PRETRAIN_FOLDER_NAME)
    clean_up(FINETUNE_FOLDER_NAME)

    print(f"Data split stored in {_SPLIT_PATH}.")


if __name__ == '__main__':
    main()
