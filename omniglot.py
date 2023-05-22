"""Dataloading for Omniglot."""
import os
import glob

import imageio
import numpy as np
import torch
from torch.utils.data import dataset, sampler, dataloader

SPLIT_PATH = os.environ['_SPLIT_PATH']
NUM_PRETRAIN_CLASSES = int(os.environ['_NUM_PRETRAIN_CLASSES'])
NUM_FINETUNE_CLASSES = int(os.environ['_NUM_FINETUNE_CLASSES'])
PRETRAIN_FOLDER_NAME = os.environ['_PRETRAIN_FOLDER_NAME']
FINETUNE_FOLDER_NAME = os.environ['_FINETUNE_FOLDER_NAME']


def load_image(file_path):
    x = imageio.imread(file_path)
    x = torch.tensor(x, dtype=torch.float32).reshape([1, 28, 28])
    x = x / 255.0
    return 1 - x


class OmniglotDataset(dataset.Dataset):
    def __init__(self, character_folders, num_images):
        """Inits OmniglotDataset.

        Args:
            num_images (int): number of examples per class
        """
        super().__init__()

        # get all character folders
        self._character_folders = character_folders

        # check problem arguments
        self._num_images = num_images

    def __getitem__(self, class_idxs):
        images_batch = []
        labels_batch = []

        for class_idx in class_idxs:
            # get a class's examples and sample from them
            all_file_paths = glob.glob(
                os.path.join(self._character_folders[class_idx], '*.png')
            )

            n_images = min(len(all_file_paths), self._num_images)
            sampled_file_paths = np.random.default_rng().choice(
                all_file_paths,
                size=n_images,
                replace=False
            )
            images = [load_image(file_path) for file_path in sampled_file_paths]

            # split sampled examples into support and query
            images_batch.extend(images)
            labels_batch.extend([class_idx] * self._num_images)

        # aggregate into tensors
        images_batch = torch.stack(images_batch)  # shape (N*S, C, H, W)
        labels_batch = torch.tensor(labels_batch)

        # shuffle batch
        batch_size = images_batch.shape[0]
        perm = np.random.default_rng().permutation(batch_size)
        images_batch = images_batch[perm]
        labels_batch = labels_batch[perm]

        return images_batch, labels_batch


class OmniglotSampler(sampler.Sampler):
    def __init__(self, split_idxs, num_classes_per_batch, num_tasks_per_epoch):
        super().__init__(None)
        self._split_idxs = split_idxs
        self._num_classes_per_batch = num_classes_per_batch
        self._num_tasks_per_epoch = num_tasks_per_epoch

    def __iter__(self):
        return (
            np.random.default_rng().choice(
                self._split_idxs,
                size=self._num_classes_per_batch,
                replace=False
            ) for _ in range(self._num_tasks_per_epoch)
        )

    def __len__(self):
        return self._num_tasks_per_epoch


def identity(x):
    return x


def get_omniglot_dataloader(
        mode,
        split,
        batch_size,
        num_classes_per_batch,
        num_images,
        num_tasks_per_epoch
):
    """Returns a dataloader.DataLoader for Omniglot.

    Args:
        mode (str): one of 'pretrain', 'finetune'
        split (str): one of 'train', 'val', 'test'
        batch_size (int): number of tasks per batch
        num_classes_per_batch (int): number of image's classes per batch
        num_images (int): number of examples per class
        num_query (int): number of query examples per class
        num_tasks_per_epoch (int): number of tasks before DataLoader is
            exhausted
    """

    if mode == 'pretrain':
        character_folders = glob.glob(
            os.path.join(SPLIT_PATH, PRETRAIN_FOLDER_NAME, split, '*/')
        )
        split_idxs = range(NUM_PRETRAIN_CLASSES)
    elif mode == 'finetune':
        character_folders = glob.glob(
            os.path.join(SPLIT_PATH, FINETUNE_FOLDER_NAME, split, '*/')
        )
        split_idxs = range(NUM_FINETUNE_CLASSES)
    else:
        raise ValueError
    
    return dataloader.DataLoader(
        dataset=OmniglotDataset(character_folders, num_images),
        batch_size=batch_size,
        sampler=OmniglotSampler(split_idxs, num_classes_per_batch, num_tasks_per_epoch),
        collate_fn=identity,
        pin_memory=torch.cuda.is_available(),
        drop_last=True
    )
