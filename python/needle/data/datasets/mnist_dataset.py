from typing import List, Optional
from ..data_basic import Dataset
import numpy as np
import gzip
import struct

class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        super(MNISTDataset, self).__init__(transforms)

        with gzip.open(image_filename, 'rb') as img_file:
            # '>' as big-endian, 'I' as unsigned int
            magic_number, num_images, num_rows, num_cols = struct.unpack('>IIII', img_file.read(16))
            
            image_data = img_file.read(num_images * num_rows * num_cols)
            self.X = np.frombuffer(image_data, dtype=np.uint8).astype(np.float32)
            # Note: Should reshape to (num_images, num_rows, num_cols, 1) in this assignment
            self.X = self.X.reshape(num_images, num_rows, num_cols, 1)
            self.X = self.X / 255.0  # Normalize to [0, 1]
        
        with gzip.open(label_filename, 'rb') as lbl_file:
            magic_number, num_labels = struct.unpack('>II', lbl_file.read(8))
            
            label_data = lbl_file.read(num_labels)
            self.y = np.frombuffer(label_data, dtype=np.uint8)

    def __getitem__(self, index) -> object:
        return self.apply_transforms(self.X[index]), self.y[index]

    def __len__(self) -> int:
        return len(self.X)