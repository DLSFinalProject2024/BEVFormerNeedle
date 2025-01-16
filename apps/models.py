import sys
sys.path.append('./python')
import needle as ndl
import needle.nn as nn
import math
import numpy as np
np.random.seed(0)

class ResNet9(ndl.nn.Module):
    def __init__(self, device=None, dtype="float32"):
        super().__init__()
        ### BEGIN YOUR SOLUTION ###
        self.device = device
        self.dtype = dtype
        self.ConvBN1 = self.ConvBN(in_channels=3, out_channels=16, kernel_size=7, stride=4)
        self.ConvBN2 = self.ConvBN(in_channels=16, out_channels=32, kernel_size=3, stride=2)
        self.Resid1  = self.ResidualBlock(in_channels=32, out_channels=32, kernel_size=3, stride=1)
        #self.Resid2  = self.ResidualBlock(in_channels=128, out_channels=128, kernel_size=3, stride=1) #Fail
        self.ConvBN3 = self.ConvBN(in_channels=32, out_channels=64, kernel_size=3, stride=2)
        #self.Resid2  = self.ResidualBlock(in_channels=128, out_channels=128, kernel_size=3, stride=1)
        self.ConvBN4 = self.ConvBN(in_channels=64, out_channels=128, kernel_size=3, stride=2)
        self.Resid2  = self.ResidualBlock(in_channels=128, out_channels=128, kernel_size=3, stride=1)
        self.Flatten = nn.Flatten()
        self.Linear1 = nn.Linear(in_features=128, out_features=128, device=self.device, dtype=self.dtype)
        self.ReLU    = nn.ReLU()
        self.Linear2 = nn.Linear(in_features=128, out_features=10, device=self.device, dtype=self.dtype)
        #self.Linear3 = nn.Linear(in_features=3072, out_features=10, device=self.device, dtype=self.dtype)
        ### END YOUR SOLUTION

    def ConvBN(self, in_channels, out_channels, kernel_size, stride): #a, b, k, s
        conv_bn_layer = nn.Sequential(
            nn.Conv(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, device=self.device, dtype=self.dtype),
            nn.BatchNorm2d(dim=out_channels, device=self.device, dtype=self.dtype),
            nn.ReLU()
        )

        return conv_bn_layer

    def ResidualBlock(self, in_channels, out_channels, kernel_size, stride): #a, b, k, s
        out_module = nn.Residual(
            nn.Sequential(
                self.ConvBN(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride),
                self.ConvBN(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride)
            )
        )

        return out_module

    def forward(self, x):
        ### BEGIN YOUR SOLUTION
        x = self.ConvBN1(x)
        x = self.ConvBN2(x)
        x = self.Resid1(x)
        x = self.ConvBN3(x)
        x = self.ConvBN4(x)
        x = self.Resid2(x)
        x = self.Flatten(x)
        x = self.Linear1(x)
        x = self.ReLU(x)
        x = self.Linear2(x)
        return x
        ### END YOUR SOLUTION


class LanguageModel(nn.Module):
    def __init__(self, embedding_size, output_size, hidden_size, num_layers=1,
                 seq_model='rnn', seq_len=40, device=None, dtype="float32"):
        """
        Consists of an embedding layer, a sequence model (either RNN or LSTM), and a
        linear layer.
        Parameters:
        output_size: Size of dictionary
        embedding_size: Size of embeddings
        hidden_size: The number of features in the hidden state of LSTM or RNN
        seq_model: 'rnn' or 'lstm', whether to use RNN or LSTM
        num_layers: Number of layers in RNN or LSTM
        """
        super(LanguageModel, self).__init__()
        ### BEGIN YOUR SOLUTION
        self.embedding_size = embedding_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.seq_model = seq_model
        self.seq_len = seq_len
        self.device = device
        self.dtype = dtype

        self.EmbeddingLayer = nn.Embedding(num_embeddings=output_size, embedding_dim=embedding_size, device=device, dtype=dtype)
        if seq_model == 'rnn':
            self.SeqModel = nn.RNN(input_size=embedding_size, hidden_size=hidden_size, num_layers=num_layers, bias=True, nonlinearity='tanh', device=device, dtype=dtype)
        elif seq_model == 'lstm':
            self.SeqModel = nn.LSTM(input_size=embedding_size, hidden_size=hidden_size, num_layers=num_layers, bias=True, device=device, dtype=dtype)
        else:
            raise ValueError("Do not support seq_model other than 'rnn' and 'lstm'.")

        self.LinearLayer = nn.Linear(in_features=hidden_size, out_features=output_size, bias=True, device=device, dtype=dtype)
        ### END YOUR SOLUTION

    def forward(self, x, h=None):
        """
        Given sequence (and the previous hidden state if given), returns probabilities of next word
        (along with the last hidden state from the sequence model).
        Inputs:
        x of shape (seq_len, bs)
        h of shape (num_layers, bs, hidden_size) if using RNN,
            else h is tuple of (h0, c0), each of shape (num_layers, bs, hidden_size)
        Returns (out, h)
        out of shape (seq_len*bs, output_size)
        h of shape (num_layers, bs, hidden_size) if using RNN,
            else h is tuple of (h0, c0), each of shape (num_layers, bs, hidden_size)
        """
        ### BEGIN YOUR SOLUTION
        seq_len, bs = x.shape
        x = self.EmbeddingLayer(x)
        out_hidden_st, h = self.SeqModel(x, h)
        out_hidden_reshape = ndl.ops.reshape(out_hidden_st, (seq_len*bs, self.hidden_size))
        out_x = self.LinearLayer(out_hidden_reshape)
        return out_x, h
        ### END YOUR SOLUTION


if __name__ == "__main__":
    model = ResNet9()
    x = ndl.ops.randu((1, 32, 32, 3), requires_grad=True)
    model(x)
    cifar10_train_dataset = ndl.data.CIFAR10Dataset("data/cifar-10-batches-py", train=True)
    train_loader = ndl.data.DataLoader(cifar10_train_dataset, 128, ndl.cpu(), dtype="float32")
    print(cifar10_train_dataset[1][0].shape)
