from typing import List, Optional
from ..data_basic import Dataset
import numpy as np
import gzip
import struct

def readTrainXImage(image_filename):
    with gzip.open(image_filename,'rb') as f:
        # Read header
        magic_number = struct.unpack('>I', f.read(4))[0]
        if magic_number != 2051:
            raise ValueError(f"Error: magic number of input image file: {image_filename} is wrong.")
        num_of_images = struct.unpack('>I', f.read(4))[0]
        rows = struct.unpack('>I', f.read(4))[0]
        cols = struct.unpack('>I', f.read(4))[0]

        # Read data
        # Pre-allocate the data array
        X = np.zeros((num_of_images, rows*cols), dtype=np.float32)

        # Read data for each image
        for i in range(num_of_images):
            #image_data = struct.unpack(f'>{rows * cols}B', f.read(rows * cols))
            # Read the exact number of bytes expected for one image
            image_bytes = f.read(rows * cols)
            if image_bytes is None:
                raise ValueError(f"Error: Read returned None for image {i}")
        
            # Debugging: Check if we got the correct number of bytes
            if len(image_bytes) != rows * cols:
                raise ValueError(f"Error reading image {i}: expected {rows * cols} bytes, got {len(image_bytes)} bytes.")            

            image_data = struct.unpack(f'>{rows * cols}B', image_bytes)
            X[i] = np.array(image_data, dtype=np.float32)
            X[i] /= 255
    return X, magic_number, num_of_images, rows, cols

def readTrainYLabel(label_filename):
    with gzip.open(label_filename,'rb') as f:
        # Read header
        magic_number = struct.unpack('>I', f.read(4))[0]
        if magic_number != 2049:
            raise ValueError(f"Error: magic number of input image file: {label_filename} is wrong.")
        num_of_items = struct.unpack('>I', f.read(4))[0]

        # Read data
        # Pre-allocate the data array
        y = np.zeros((num_of_items), dtype=np.uint8)

        for i in range(num_of_items):
            label_bytes = f.read(1)
            if label_bytes is None:
                raise ValueError(f"Error: Read returned None for item {i}")
        
            # Debugging: Check if we got the correct number of bytes
            if len(label_bytes) != 1:
                raise ValueError(f"Error reading item {i}: expected 1 bytes, got {len(label_bytes)} bytes.")            

            label_data = struct.unpack(f'>B', label_bytes)
            y[i] = np.array(label_data, dtype=np.uint8)[0]
    return y, magic_number, num_of_items


class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        '''
        Make sure return X: HxWxC, y: 1D array containing labels of the exmaples.
        
        Tuple (X,y):
            X (numpy.ndarray[np.float32]): 2D numpy array containing the loaded
                data.  The dimensionality of the data should be
                (num_examples x input_dim) where 'input_dim' is the full
                dimension of the data, e.g., since MNIST images are 28x28, it
                will be 784.  Values should be of type np.float32, and the data
                should be normalized to have a minimum value of 0.0 and a
                maximum value of 1.0.

            y (numpy.ndarray[dypte=np.int8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.int8 and
                for MNIST will contain the values 0-9.
        '''
        ### BEGIN YOUR SOLUTION
        super().__init__(transforms)
        X, magic_number_trainX, num_of_images, rows, cols = readTrainXImage(image_filename)
        y, magic_number_trainY, num_of_items = readTrainYLabel(label_filename) 
        self.num_of_items = num_of_items
        self.X            = X
        self.y            = y
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        ### BEGIN YOUR SOLUTION
        if isinstance(index, slice):
            selected_X = self.X[index]
            selected_y = self.y[index]
        
            # If transforms are defined, apply them to each item in the slice
            if self.transforms:
                X_transformed = [self.apply_transforms(x.reshape((28, 28, -1))).reshape((28, 28, -1)) for x in selected_X]
                #X_transformed = [self.apply_transforms(x.reshape((28, 28, -1))).reshape((784,)) for x in selected_X]
                return np.array(X_transformed), np.array(selected_y)
            else:
                # Return the sliced data as reshaped arrays
                return selected_X.reshape(-1, 28, 28, 1), selected_y            
                #return selected_X.reshape((784,)), selected_y            
        else:
            selected_X = self.X[index]
            selected_y = self.y[index]
            if self.transforms:
                X_transformed = self.apply_transforms(selected_X.reshape((28, 28, -1))).reshape((28, 28, -1))
                #X_transformed = self.apply_transforms(selected_X.reshape((28, 28, -1))).reshape((784,))
                return X_transformed, selected_y
            else:
                return selected_X.reshape((28, 28, -1)), selected_y
                #return selected_X.reshape((784,)), selected_y

        ### END YOUR SOLUTION

    def __len__(self) -> int:
        ### BEGIN YOUR SOLUTION
        return self.num_of_items
        ### END YOUR SOLUTION