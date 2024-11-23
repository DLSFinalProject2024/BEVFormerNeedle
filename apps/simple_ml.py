"""hw1/apps/simple_ml.py"""

import struct
import gzip
import numpy as np

import sys

sys.path.append("python/")
import needle as ndl

import needle.nn as nn
from apps.models import *
import time
device = ndl.cpu()

def parse_mnist(image_filename, label_filename):
    """Read an images and labels file in MNIST format.  See this page:
    http://yann.lecun.com/exdb/mnist/ for a description of the file format.

    Args:
        image_filename (str): name of gzipped images file in MNIST format
        label_filename (str): name of gzipped labels file in MNIST format

    Returns:
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
    """
    with gzip.open(image_filename, 'rb') as img_file:
        # '>' as big-endian, 'I' as unsigned int
        magic_number, num_images, num_rows, num_cols = struct.unpack('>IIII', img_file.read(16))
        
        image_data = img_file.read(num_images * num_rows * num_cols)
        X = np.frombuffer(image_data, dtype=np.uint8)
        X = X.reshape(num_images, num_rows * num_cols).astype(np.float32)
        X = X / 255.0  # Normalize to [0, 1]
    
    with gzip.open(label_filename, 'rb') as lbl_file:
        magic_number, num_labels = struct.unpack('>II', lbl_file.read(8))
        
        label_data = lbl_file.read(num_labels)
        y = np.frombuffer(label_data, dtype=np.uint8)
    
    return X, y


def softmax_loss(Z: ndl.Tensor, y_one_hot: ndl.Tensor):
    """Return softmax loss.  Note that for the purposes of this assignment,
    you don't need to worry about "nicely" scaling the numerical properties
    of the log-sum-exp computation, but can just compute this directly.

    Args:
        Z (ndl.Tensor[np.float32]): 2D Tensor of shape
            (batch_size, num_classes), containing the logit predictions for
            each class.
        y (ndl.Tensor[np.int8]): 2D Tensor of shape (batch_size, num_classes)
            containing a 1 at the index of the true label of each example and
            zeros elsewhere.

    Returns:
        Average softmax loss over the sample. (ndl.Tensor[np.float32])
    """
    logsumexp = ndl.log(ndl.summation(ndl.exp(Z), axes=(1,)))
    Zy = ndl.summation(Z * y_one_hot, axes=(1,))
    return ndl.summation(logsumexp - Zy, axes=(0,)) / Z.shape[0]


def nn_epoch(X, y, W1, W2, lr=0.1, batch=100):
    """Run a single epoch of SGD for a two-layer neural network defined by the
    weights W1 and W2 (with no bias terms):
        logits = ReLU(X * W1) * W1
    The function should use the step size lr, and the specified batch size (and
    again, without randomizing the order of X).

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        W1 (ndl.Tensor[np.float32]): 2D array of first layer weights, of shape
            (input_dim, hidden_dim)
        W2 (ndl.Tensor[np.float32]): 2D array of second layer weights, of shape
            (hidden_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD mini-batch

    Returns:
        Tuple: (W1, W2)
            W1: ndl.Tensor[np.float32]
            W2: ndl.Tensor[np.float32]
    """
    num_examples = X.shape[0]
    k = W2.shape[-1] # num_classes (output_dim)

    for i in range(0, num_examples, batch):
        X_batch = ndl.Tensor(X[i:i + batch])
        y_batch = y[i:i + batch]

        # Create one-hot encoded ndl.tensor for y_batch, shape (batch_size, k)
        y_one_hot = np.zeros((batch, k))
        y_one_hot[np.arange(batch), y_batch] = 1
        y_batch = ndl.Tensor(y_one_hot)

        # Forward
        Z = ndl.relu(X_batch @ W1) @ W2

        loss = softmax_loss(Z, y_batch)
        loss.backward()
        
        # Note: Since we did not implement optimizer.zero_grad() like Pytorch,
        # we need to transform gradients to numpy and then back to ndl.Tensor
        # to avoid accumulating gradients
        W1 -= lr * W1.grad.detach()
        W2 -= lr * W2.grad.detach()
        # Note: numpy also works
        # W1 = ndl.Tensor(W1.numpy() - lr * W1.grad.numpy())
        # W2 = ndl.Tensor(W2.numpy() - lr * W2.grad.numpy())

    return W1, W2

### CIFAR-10 training ###
def epoch_general_cifar10(dataloader, model, loss_fn=nn.SoftmaxLoss(), opt=None):
    """
    Iterates over the dataloader. If optimizer is not None, sets the
    model to train mode, and for each batch updates the model parameters.
    If optimizer is None, sets the model to eval mode, and simply computes
    the loss/accuracy.

    Args:
        dataloader: Dataloader instance
        model: nn.Module instance
        loss_fn: nn.Module instance
        opt: Optimizer instance (optional)

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    raise NotImplementedError()
    ### END YOUR SOLUTION


def train_cifar10(model, dataloader, n_epochs=1, optimizer=ndl.optim.Adam,
          lr=0.001, weight_decay=0.001, loss_fn=nn.SoftmaxLoss):
    """
    Performs {n_epochs} epochs of training.

    Args:
        dataloader: Dataloader instance
        model: nn.Module instance
        n_epochs: number of epochs (int)
        optimizer: Optimizer class
        lr: learning rate (float)
        weight_decay: weight decay (float)
        loss_fn: nn.Module class

    Returns:
        avg_acc: average accuracy over dataset from last epoch of training
        avg_loss: average loss over dataset from last epoch of training
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    raise NotImplementedError()
    ### END YOUR SOLUTION


def evaluate_cifar10(model, dataloader, loss_fn=nn.SoftmaxLoss):
    """
    Computes the test accuracy and loss of the model.

    Args:
        dataloader: Dataloader instance
        model: nn.Module instance
        loss_fn: nn.Module class

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    raise NotImplementedError()
    ### END YOUR SOLUTION


### PTB training ###
def epoch_general_ptb(data, model, seq_len=40, loss_fn=nn.SoftmaxLoss(), opt=None,
        clip=None, device=None, dtype="float32"):
    """
    Iterates over the data. If optimizer is not None, sets the
    model to train mode, and for each batch updates the model parameters.
    If optimizer is None, sets the model to eval mode, and simply computes
    the loss/accuracy.

    Args:
        data: data of shape (nbatch, batch_size) given from batchify function
        model: LanguageModel instance
        seq_len: i.e. bptt, sequence length
        loss_fn: nn.Module instance
        opt: Optimizer instance (optional)
        clip: max norm of gradients (optional)

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    np.random.seed(4)

    model.eval() if opt is None else model.train()
    nbatch, batch_size = data.shape
    errs = []
    losses = []
    h_prev = None

    for i in range(nbatch - seq_len):
        X, y = ndl.data.get_batch(data, i, seq_len, device=device, dtype=dtype)
        output, h_prev = model(X, h_prev)
        loss = loss_fn(output, y)
        
        if model.training:
            opt.reset_grad()
            loss.backward()
            if clip:  # Optional, not implemented
                nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()

        errs.append((output.numpy().argmax(1) != y.numpy()).sum().item())
        losses.append(loss.numpy().item())

        # Detach hidden states to prevent backprop through entire sequence
        if isinstance(h_prev, tuple):  # LSTM
            h, c = h_prev
            h_prev = (h.detach(), c.detach())
        else:  # RNN
            h_prev = h_prev.detach()

    avg_acc = 1 - sum(errs) / (nbatch - seq_len)
    avg_loss = sum(losses) / (nbatch - seq_len)
    return avg_acc, avg_loss


def train_ptb(model, data, seq_len=40, n_epochs=1, optimizer=ndl.optim.SGD,
          lr=4.0, weight_decay=0.0, loss_fn=nn.SoftmaxLoss, clip=None,
          device=None, dtype="float32"):
    """
    Performs {n_epochs} epochs of training.

    Args:
        model: LanguageModel instance
        data: data of shape (nbatch, batch_size) given from batchify function
        seq_len: i.e. bptt, sequence length
        n_epochs: number of epochs (int)
        optimizer: Optimizer class
        lr: learning rate (float)
        weight_decay: weight decay (float)
        loss_fn: nn.Module class
        clip: max norm of gradients (optional)

    Returns:
        avg_acc: average accuracy over dataset from last epoch of training
        avg_loss: average loss over dataset from last epoch of training
    """
    np.random.seed(4)
    
    opt = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)
    for epoch in range(n_epochs):
        avg_acc, avg_loss = epoch_general_ptb(data, model, seq_len=seq_len, loss_fn=loss_fn(),
            opt=opt, clip=clip, device=device, dtype=dtype)
        
        # print(f"Epoch {epoch+1}: avg_acc={avg_acc:.3f}, avg_loss={avg_loss:.3f}")
    
    # return np.array(avg_acc), np.array(avg_loss)
    return avg_acc, avg_loss

def evaluate_ptb(model, data, seq_len=40, loss_fn=nn.SoftmaxLoss,
        device=None, dtype="float32"):
    """
    Computes the test accuracy and loss of the model.

    Args:
        model: LanguageModel instance
        data: data of shape (nbatch, batch_size) given from batchify function
        seq_len: i.e. bptt, sequence length
        loss_fn: nn.Module class

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    np.random.seed(4)
    
    avg_acc, avg_loss = epoch_general_ptb(data, model, seq_len=seq_len, loss_fn=loss_fn(),
        opt=None, device=device, dtype=dtype)
    return avg_acc, avg_loss

### CODE BELOW IS FOR ILLUSTRATION, YOU DO NOT NEED TO EDIT


def loss_err(h, y):
    """Helper function to compute both loss and error"""
    y_one_hot = np.zeros((y.shape[0], h.shape[-1]))
    y_one_hot[np.arange(y.size), y] = 1
    y_ = ndl.Tensor(y_one_hot)
    return softmax_loss(h, y_).numpy(), np.mean(h.numpy().argmax(axis=1) != y)
