import numpy as np
from ..autograd import Tensor

from typing import Iterator, Optional, List, Sized, Union, Iterable, Any



class Dataset:
    r"""An abstract class representing a `Dataset`.

    All subclasses should overwrite :meth:`__getitem__`, supporting fetching a
    data sample for a given key. Subclasses must also overwrite
    :meth:`__len__`, which is expected to return the size of the dataset.
    """

    def __init__(self, transforms: Optional[List] = None):
        self.transforms = transforms

    def __getitem__(self, index) -> object:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError
    
    def apply_transforms(self, x):
        if self.transforms is not None:
            # apply the transforms
            for tform in self.transforms:
                x = tform(x)
        return x


class DataLoader:
    r"""
    Data loader. Combines a dataset and a sampler, and provides an iterable over
    the given dataset.
    Args:
        dataset (Dataset): dataset from which to load the data.
        batch_size (int, optional): how many samples per batch to load
            (default: ``1``).
        shuffle (bool, optional): set to ``True`` to have the data reshuffled
            at every epoch (default: ``False``).
     """
    dataset: Dataset
    batch_size: Optional[int]

    def __init__(
        self,
        dataset: Dataset,
        batch_size: Optional[int] = 1,
        shuffle: bool = False,
    ):

        self.batch_idx = 0
        self.dataset = dataset
        self.shuffle = shuffle
        self.batch_size = batch_size
        if not self.shuffle:
            self.ordering = np.array_split(np.arange(len(dataset)), 
                                           range(batch_size, len(dataset), batch_size))

    def __iter__(self):
        ### BEGIN YOUR SOLUTION
        if self.shuffle:
            self.ordering = np.random.permutation(len(self.dataset))
        else:
            self.ordering = np.arange(len(self.dataset))

        self.batches_order = np.array_split(self.ordering, 
                                            range(self.batch_size, len(self.dataset), self.batch_size))

        self.batch_idx = 0
        ### END YOUR SOLUTION
        return self

    def __next__(self):
        ### BEGIN YOUR SOLUTION
        if self.batch_idx >= len(self.batches_order):
            raise StopIteration

        current_batch = [self.dataset[x] for x in self.batches_order[self.batch_idx]]
        current_batch_tensor = [
            tuple(item for item in sample)
            if len(sample) > 1 else sample
            for sample in current_batch
        ]
        final_output = [] #n-tuple output
        item_list = {}
        one_list = []

        for sample in current_batch_tensor:
            if isinstance(sample, tuple):
                for id, item in enumerate(sample):
                    if id not in item_list:
                        item_list[id] = [item]
                    else:
                        item_list[id].append(item)
            else:
                one_list.append(sample)

        if len(one_list) == 0:
            for key, ele in item_list.items():
                final_output.append(Tensor(np.array(ele)))
        else:
            final_output.append(Tensor(np.array(one_list)))
        
        self.batch_idx += 1
        #tensor_batch = Tensor(current_batch)
        return tuple(final_output)
        ### END YOUR SOLUTION

