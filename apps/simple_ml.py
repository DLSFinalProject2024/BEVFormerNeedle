"""hw1/apps/simple_ml.py"""

import struct
import gzip
import numpy as np
from tqdm import tqdm

import sys

sys.path.append("python/")
import needle as ndl

import needle.nn as nn
from apps.models import *
from needle.data import DataLoader, CIFAR10Dataset
import time
device = ndl.cpu()
#device = ndl.cpu() if not ndl.cuda().enabled() else ndl.cuda()

import psutil
import os

def print_memory_usage(message=""):
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    mem_usage = mem_info.rss / (1024 ** 2)
    print(f"[{message}] Memory usage: {mem_usage:.2f} MB")  # RSS: Resident Set Size
    return mem_usage

import threading
import os
import time

def monitor_memory_usage():
    process = psutil.Process(os.getpid())
    while True:
        mem_info = process.memory_info()
        print(f"Memory usage: {mem_info.rss / (1024 ** 2):.2f} MB")
        time.sleep(0.05)  # Adjust the sleep interval as needed

def parse_mnist(image_filesname, label_filename):
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
    ### BEGIN YOUR SOLUTION
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
    ### END YOUR SOLUTION


def softmax_loss(Z, y_one_hot):
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
    ### BEGIN YOUR SOLUTION
    logsumexp = ndl.log(ndl.summation(ndl.exp(Z), axes=(1,)))
    Zy = ndl.summation(Z * y_one_hot, axes=(1,))
    return ndl.summation(logsumexp - Zy, axes=(0,)) / Z.shape[0]
    ### END YOUR SOLUTION


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

    ### BEGIN YOUR SOLUTION
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
    ### END YOUR SOLUTION

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
    model.eval() if opt is None else model.train()

    total_loss = 0
    correct = 0

    for batch_data, batch_labels in tqdm(dataloader):
        batch_data = ndl.Tensor(batch_data, device=device)
        batch_labels = ndl.Tensor(batch_labels, device=device)
        
        batch_size = batch_data.shape[0]
        if model.training:
            opt.reset_grad()

        logits = model(batch_data)
        loss = loss_fn(logits, batch_labels)

        if model.training:
            loss.backward()
            opt.step()

        # Compute metrics
        total_loss += loss.detach().numpy() * batch_size
        correct += (logits.detach().numpy().argmax(axis=1) == batch_labels.detach().numpy()).sum()

    avg_loss = total_loss / len(dataloader.dataset)
    avg_acc = correct / len(dataloader.dataset)

    return avg_acc.item(), avg_loss.item()
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
    
    opt = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)
    avg_acc_list = []
    avg_loss_list = []

    for epoch in range(n_epochs):
        avg_acc, avg_loss = epoch_general_cifar10(dataloader, model, loss_fn=loss_fn(), opt=opt)
        print(f"Training, Epoch {epoch+1}: avg_acc={avg_acc:.3f}, avg_loss={avg_loss:.3f}")
        avg_acc_list.append(avg_acc)
        avg_loss_list.append(avg_loss)

    return avg_acc_list, avg_loss_list
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
    avg_acc, avg_loss = epoch_general_cifar10(dataloader, model, loss_fn=loss_fn(), opt=None)
    print(f"Testing, avg_acc={avg_acc:.3f}, avg_loss={avg_loss:.3f}")
    return avg_acc, avg_loss
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
    ### BEGIN YOUR SOLUTION
    nbatch, batch_size = data.shape
    avg_loss = []
    avg_acc  = []
    sample_num = 0
    if opt is not None:
        # Training Mode
        model.train()
        h = None
        for i in range(0, nbatch-1, seq_len):
            opt.reset_grad()
            batch_x, batch_y = ndl.data.get_batch(batches=data, i=i, bptt=seq_len, device=device, dtype=dtype)
            out_y, h = model(batch_x, h)

            # h is not updated grad in further sequence for saving memory
            if isinstance(h, tuple):
                hi_det_list = []
                for hi in h:
                    hi_det_list.append(hi.detach())
                h = tuple([_ for _ in hi_det_list])
            else:
                h = h.detach()

            loss = loss_fn(out_y, batch_y)
            loss.backward()
            opt.step()
            avg_loss.append(np.float32(loss.numpy())*batch_y.shape[0])
            avg_acc.append(np.float32(np.sum(batch_y.numpy() == out_y.numpy().argmax(axis=1))))
            sample_num += batch_y.shape[0]
    else:
        # Testing Mode
        model.eval()
        h = None
        for i in range(0, nbatch-1, seq_len):
            batch_x, batch_y = ndl.data.get_batch(batches=data, i=i, bptt=seq_len, device=device, dtype=dtype)
            out_y, h = model(batch_x, h)

            # h is not updated grad in further sequence for saving memory
            if isinstance(h, tuple):
                hi_det_list = []
                for hi in h:
                    hi_det_list.append(hi.detach())
                h = tuple([_ for _ in hi_det_list])
            else:
                h = h.detach()

            loss = loss_fn(out_y, batch_y)
            avg_loss.append(np.float32(loss.numpy()*batch_y.shape[0]))
            avg_acc.append(np.float32(np.sum(batch_y.numpy() == out_y.numpy().argmax(axis=1))))
            sample_num += batch_y.shape[0]

    avg_loss_val = np.sum(avg_loss)/sample_num
    avg_acc_val  = np.sum(avg_acc)/sample_num
    return avg_acc_val, avg_loss_val
    ### END YOUR SOLUTION


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
    ### BEGIN YOUR SOLUTION
    opt = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Training
    #monitor_thread = threading.Thread(target=monitor_memory_usage, daemon=True)
    #monitor_thread.start()
    for loop_i in range(n_epochs):
        avg_acc_train, avg_loss_train = epoch_general_ptb(data=data, model=model, seq_len=seq_len, loss_fn=loss_fn(), opt=opt,
                                                          clip=clip, device=device, dtype=dtype)
    return avg_acc_train, avg_loss_train
    ### END YOUR SOLUTION

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
    ### BEGIN YOUR SOLUTION
    avg_acc_test, avg_loss_test = epoch_general_ptb(data=data, model=model, seq_len=seq_len, loss_fn=loss_fn(), opt=None,
                                                    clip=None, device=device, dtype=dtype)
    return avg_acc_test, avg_loss_test
    ### END YOUR SOLUTION

### CODE BELOW IS FOR ILLUSTRATION, YOU DO NOT NEED TO EDIT


def loss_err(h, y):
    """Helper function to compute both loss and error"""
    y_one_hot = np.zeros((y.shape[0], h.shape[-1]))
    y_one_hot[np.arange(y.size), y] = 1
    y_ = ndl.Tensor(y_one_hot)
    return softmax_loss(h, y_).numpy(), np.mean(h.numpy().argmax(axis=1) != y)
