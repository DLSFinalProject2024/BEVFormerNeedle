import os
import pickle
from typing import Iterator, Optional, List, Sized, Union, Iterable, Any
import numpy as np
from ..data_basic import Dataset

class CIFAR10Dataset(Dataset):
    def __init__(
        self,
        base_folder: str,
        train: bool,
        p: Optional[int] = 0.5,
        transforms: Optional[List] = None
    ):
        """
        Parameters:
        base_folder - cifar-10-batches-py folder filepath
        train - bool, if True load training dataset, else load test dataset
        Divide pixel values by 255. so that images are in 0-1 range.
        Attributes:
        X - numpy array of images
        y - numpy array of labels
        """
        self.transforms = transforms

        train_names = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']
        test_names = ['test_batch']
        names = train_names if train else test_names
        data_batches = []
        for name in names:
            with open(os.path.join(base_folder, name), 'rb') as f:
                batch = pickle.load(f, encoding='bytes')
                data_batches.append(batch)
        
        # batch.keys() = [b'batch_label', b'labels', b'data', b'filenames']
        self.X = np.concatenate([batch[b'data'] for batch in data_batches])
        self.y = np.concatenate([batch[b'labels'] for batch in data_batches])

        # Transform, BCHW
        self.X = self.X.reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
        self.y = self.y.astype(np.int64)

    def __getitem__(self, index) -> object:
        """
        Returns the image, label at given index
        Image should be of shape (3, 32, 32)
        """
        return self.X[index], self.y[index]

    def __len__(self) -> int:
        """
        Returns the total number of examples in the dataset
        """
        return len(self.X)
