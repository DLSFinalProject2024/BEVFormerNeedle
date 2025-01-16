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
        ### BEGIN YOUR SOLUTION
        self.train = train
        self.p = p
        self.transforms = transforms
        self.base_folder = base_folder

        if train:
            self.batchs_names = [f"data_batch_{i}" for i in range(1, 6)]
        else:
            self.batchs_names = [f"test_batch"]

        self.X, self.y = self.load_data()
        self.X = self.X/255.0
        ### END YOUR SOLUTION

    def load_data(self):
        import pickle
        data = []
        labels = []
        for batch_name in self.batchs_names:
            with open(os.path.join(self.base_folder, batch_name), 'rb') as file:
                batch = pickle.load(file, encoding='bytes')
                data.append(batch[b'data'])
                labels.extend(batch[b'labels'])
        
        data = np.vstack(data).reshape(-1, 3, 32, 32).astype(np.float32)
        labels = np.array(labels, dtype=np.int64)
        
        return data, labels

    def __getitem__(self, index) -> object:
        """
        Returns the image, label at given index
        Image should be of shape (3, 32, 32)
        """
        ### BEGIN YOUR SOLUTION
        img, label = self.X[index], self.y[index]
        
        # Apply transformations if provided
        if self.transforms:
            for transform in self.transforms:
                img = transform(img)
        
        return img, label        
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        """
        Returns the total number of examples in the dataset
        """
        ### BEGIN YOUR SOLUTION
        return len(self.y)
        ### END YOUR SOLUTION
