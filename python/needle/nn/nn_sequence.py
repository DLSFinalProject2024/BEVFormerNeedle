"""The module.
"""
from typing import List
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np
from .nn_basic import Parameter, Module
import needle.nn as nn


class Sigmoid(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.power_scalar((1+ops.exp(-x)), -1)
        ### END YOUR SOLUTION

class RNNCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        """
        Applies an RNN cell with tanh or ReLU nonlinearity.

        Parameters:
        input_size: The number of expected features in the input X
        hidden_size: The number of features in the hidden state h
        bias: If False, then the layer does not use bias weights
        nonlinearity: The non-linearity to use. Can be either 'tanh' or 'relu'.

        Variables:
        W_ih: The learnable input-hidden weights of shape (input_size, hidden_size).
        W_hh: The learnable hidden-hidden weights of shape (hidden_size, hidden_size).
        bias_ih: The learnable input-hidden bias of shape (hidden_size,).
        bias_hh: The learnable hidden-hidden bias of shape (hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        self.device = device
        self.dtype = dtype
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.nonlinearity = nonlinearity

        bound = np.sqrt(1/hidden_size)
        self.W_ih = Parameter(init.rand(*(input_size, hidden_size), low=-bound, high=bound, dtype=dtype, device=device, requires_grad=True))
        self.W_hh = Parameter(init.rand(*(hidden_size, hidden_size), low=-bound, high=bound, dtype=dtype, device=device, requires_grad=True))
        if bias:
            self.bias_ih = Parameter(init.rand(*(hidden_size, ), low=-bound, high=bound, dtype=dtype, device=device, requires_grad=True))
            self.bias_hh = Parameter(init.rand(*(hidden_size, ), low=-bound, high=bound, dtype=dtype, device=device, requires_grad=True))
        else:
            self.bias_ih = None
            self.bias_hh = None
        
        if nonlinearity == 'tanh':
            self.nonlinear_func = nn.Tanh()
        elif nonlinearity == 'relu':
            self.nonlinear_func = nn.ReLU()
        else:
            raise ValueError("Do not support nonlinearity other than 'tanh' or 'relu' function.")

        ### END YOUR SOLUTION

    def forward(self, X, h=None):
        """
        Inputs:
        X of shape (bs, input_size): Tensor containing input features
        h of shape (bs, hidden_size): Tensor containing the initial hidden state
            for each element in the batch. Defaults to zero if not provided.

        Outputs:
        h' of shape (bs, hidden_size): Tensor contianing the next hidden state
            for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        bs, input_size = X.shape

        if h is None:
            h = init.zeros(*(bs, self.hidden_size), device=self.device, dtype=self.dtype, requires_grad=False)
        
        if self.bias:
            comb_t = (h @ self.W_hh) + (X @ self.W_ih) + self.bias_ih.reshape((1, self.hidden_size)).broadcast_to((bs, self.hidden_size)) + self.bias_hh.reshape((1, self.hidden_size)).broadcast_to((bs, self.hidden_size))
            h_t = self.nonlinear_func(comb_t)
        else:
            comb_t = (h @ self.W_hh) + (X @ self.W_ih)
            h_t = self.nonlinear_func(comb_t)

        return h_t
        ### END YOUR SOLUTION


class RNN(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        """
        Applies a multi-layer RNN with tanh or ReLU non-linearity to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        nonlinearity - The non-linearity to use. Can be either 'tanh' or 'relu'.
        bias - If False, then the layer does not use bias weights.

        Variables:
        rnn_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, hidden_size) for k=0. Otherwise the shape is
            (hidden_size, hidden_size).
        rnn_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, hidden_size).
        rnn_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (hidden_size,).
        rnn_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (hidden_size,).
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.nonlinearity = nonlinearity
        self.device = device
        self.dtype = dtype

        self.rnn_cells = []
        self.rnn_cells.append(RNNCell(input_size, hidden_size, bias, nonlinearity, device, dtype))
        for i in range(num_layers-1): # upper layers
            self.rnn_cells.append(RNNCell(hidden_size, hidden_size, bias, nonlinearity, device, dtype))
        ### END YOUR SOLUTION

    def forward(self, X, h0=None):
        """
        Inputs:
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h_0 of shape (num_layers, bs, hidden_size) containing the initial
            hidden state for each element in the batch. Defaults to zeros if not provided.

        Outputs
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the RNN, for each t.
        h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        seq_len, bs, input_size = X.shape

        if h0 is None:
            h0 = [None for _ in range(self.num_layers)]
        else:
            h0 = list(ops.split(h0, axis=0)) #h0 = [h0_layer1, h0_layer2, ..., h0_layerN]

        h_list = [hidden_st for hidden_st in h0]
        input_seq_X = list(ops.split(X, axis=0)) #input_seq_X = [X0, X1, X2, ..., Xt]
        out_list = [] #output result for seq_len sequences

        for t in range(seq_len):
            h_applied_layer = [hidden_st for hidden_st in h_list]
            input_X = input_seq_X[t]
            for layer in range(self.num_layers):
                h_out_layer = self.rnn_cells[layer](input_X, h_applied_layer[layer])
                input_X = h_out_layer # next layer input
                h_list[layer] = h_out_layer # next sequence hidden state input

            out_list.append(input_X)
        
        out_hidden_st = ops.stack(out_list, axis=0)
        h_n = ops.stack(h_list, axis=0)

        return out_hidden_st, h_n
        ### END YOUR SOLUTION


class LSTMCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, device=None, dtype="float32"):
        """
        A long short-term memory (LSTM) cell.

        Parameters:
        input_size - The number of expected features in the input X
        hidden_size - The number of features in the hidden state h
        bias - If False, then the layer does not use bias weights

        Variables:
        W_ih - The learnable input-hidden weights, of shape (input_size, 4*hidden_size).
        W_hh - The learnable hidden-hidden weights, of shape (hidden_size, 4*hidden_size).
        bias_ih - The learnable input-hidden bias, of shape (4*hidden_size,).
        bias_hh - The learnable hidden-hidden bias, of shape (4*hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        self.input_size  = input_size
        self.hidden_size = hidden_size
        self.bias   = bias
        self.device = device
        self.dtype  = dtype

        bound = np.sqrt(1/hidden_size)
        self.W_ih = Parameter(init.rand(*(input_size, 4*hidden_size), low=-bound, high=bound, dtype=dtype, device=device, requires_grad=True))
        self.W_hh = Parameter(init.rand(*(hidden_size, 4*hidden_size), low=-bound, high=bound, dtype=dtype, device=device, requires_grad=True))
        if bias:
            self.bias_ih = Parameter(init.rand(*(4*hidden_size, ), low=-bound, high=bound, dtype=dtype, device=device, requires_grad=True))
            self.bias_hh = Parameter(init.rand(*(4*hidden_size, ), low=-bound, high=bound, dtype=dtype, device=device, requires_grad=True))
        else:
            self.bias_ih = None
            self.bias_hh = None

        self.sigmoid_i = Sigmoid()
        self.sigmoid_f = Sigmoid()
        self.tanh_g    = nn.Tanh()
        self.sigmoid_o = Sigmoid()
        self.tanh_o    = nn.Tanh()
        ### END YOUR SOLUTION


    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (batch, input_size): Tensor containing input features
        h, tuple of (h0, c0), with
            h0 of shape (bs, hidden_size): Tensor containing the initial hidden state
                for each element in the batch. Defaults to zero if not provided.
            c0 of shape (bs, hidden_size): Tensor containing the initial cell state
                for each element in the batch. Defaults to zero if not provided.

        Outputs: (h', c')
        h' of shape (bs, hidden_size): Tensor containing the next hidden state for each
            element in the batch.
        c' of shape (bs, hidden_size): Tensor containing the next cell state for each
            element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        bs, input_size = X.shape

        if h is None:
            h0 = init.zeros(*(bs, self.hidden_size), device=self.device, dtype=self.dtype, requires_grad=False)
            c0 = init.zeros(*(bs, self.hidden_size), device=self.device, dtype=self.dtype, requires_grad=False)
        else:
            h0, c0 = h
            if h0 is None:
                h0 = init.zeros(*(bs, self.hidden_size), device=self.device, dtype=self.dtype, requires_grad=False)
            if c0 is None:
                c0 = init.zeros(*(bs, self.hidden_size), device=self.device, dtype=self.dtype, requires_grad=False)

        if self.bias:
            comb_t = (h0 @ self.W_hh) + (X @ self.W_ih) + self.bias_ih.reshape((1, 4*self.hidden_size)).broadcast_to((bs, 4*self.hidden_size)) + self.bias_hh.reshape((1, 4*self.hidden_size)).broadcast_to((bs, 4*self.hidden_size))
        else:
            comb_t = (h0 @ self.W_hh) + (X @ self.W_ih)

        comb_t_split = ops.split(comb_t, axis=1) #[bs x 1, bs x 2, ..., bs x hidden_size,..., bs x (hidden_size*2), ..., bs x (hidden_size*4)]
        i_ = ops.stack([comb_t_split[i] for i in range(0                 , self.hidden_size)], axis=1)
        f_ = ops.stack([comb_t_split[i] for i in range(self.hidden_size  , self.hidden_size*2)], axis=1)
        g_ = ops.stack([comb_t_split[i] for i in range(self.hidden_size*2, self.hidden_size*3)], axis=1)
        o_ = ops.stack([comb_t_split[i] for i in range(self.hidden_size*3, self.hidden_size*4)], axis=1)
        i = self.sigmoid_i(i_)
        f = self.sigmoid_f(f_)
        g = self.tanh_g(g_)
        o = self.sigmoid_o(o_)

        c_out = f*c0 + i*g
        h_out = o*self.tanh_o(c_out)

        return h_out, c_out
        ### END YOUR SOLUTION


class LSTM(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        """
        Applies a multi-layer long short-term memory (LSTM) RNN to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        bias - If False, then the layer does not use bias weights.

        Variables:
        lstm_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, 4*hidden_size) for k=0. Otherwise the shape is
            (hidden_size, 4*hidden_size).
        lstm_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, 4*hidden_size).
        lstm_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        lstm_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        """
        ### BEGIN YOUR SOLUTION
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.device = device
        self.dtype = dtype

        self.lstm_cells = []
        self.lstm_cells.append(LSTMCell(input_size, hidden_size, bias, device, dtype))
        for i in range(num_layers-1): # upper layers
            self.lstm_cells.append(LSTMCell(hidden_size, hidden_size, bias, device, dtype))


        ### END YOUR SOLUTION

    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h, tuple of (h0, c0) with
            h_0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden state for each element in the batch. Defaults to zeros if not provided.
            c0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden cell state for each element in the batch. Defaults to zeros if not provided.

        Outputs: (output, (h_n, c_n))
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the LSTM, for each t.
        tuple of (h_n, c_n) with
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden cell state for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        seq_len, bs, input_size = X.shape

        if h is None:
            h0 = [None for _ in range(self.num_layers)]
            c0 = [None for _ in range(self.num_layers)]
        else:
            h0, c0 = h
            if h0 is None:
                h0 = [None for _ in range(self.num_layers)]
            else:
                h0 = list(ops.split(h0, axis=0)) #h0 = [h0_layer1, h0_layer2, ..., h0_layerN]

            if c0 is None:
                c0 = [None for _ in range(self.num_layers)]
            else:
                c0 = list(ops.split(c0, axis=0)) #c0 = [c0_layer1, c0_layer2, ..., c0_layerN]

        h_list = [hidden_st for hidden_st in h0]
        c_list = [hidden_st for hidden_st in c0]
        input_seq_X = list(ops.split(X, axis=0)) #input_seq_X = [X0, X1, X2, ..., Xt]
        out_list = [] #output result for seq_len sequences

        for t in range(seq_len):
            h_applied_layer = [hidden_st for hidden_st in h_list]
            c_applied_layer = [hidden_st for hidden_st in c_list]
            input_X = input_seq_X[t]
            for layer in range(self.num_layers):
                h_out_layer, c_out_layer = self.lstm_cells[layer](input_X, (h_applied_layer[layer], c_applied_layer[layer]))
                input_X = h_out_layer # next layer input
                h_list[layer] = h_out_layer # next sequence hidden state input
                c_list[layer] = c_out_layer # next sequence hidden state input

            out_list.append(input_X)
        
        out_hidden_st = ops.stack(out_list, axis=0)
        h_n = ops.stack(h_list, axis=0)
        c_n = ops.stack(c_list, axis=0)

        return out_hidden_st, (h_n, c_n)
        ### END YOUR SOLUTION

class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype="float32"):
        super().__init__()
        """
        Maps one-hot word vectors from a dictionary of fixed size to embeddings.

        Parameters:
        num_embeddings (int) - Size of the dictionary
        embedding_dim (int) - The size of each embedding vector

        Variables:
        weight - The learnable weights of shape (num_embeddings, embedding_dim)
            initialized from N(0, 1).
        """
        ### BEGIN YOUR SOLUTION
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.device = device
        self.dtype = dtype
        self.weight = Parameter(init.randn(*(num_embeddings, embedding_dim), mean=0, std=1, device=device, dtype=dtype, requires_grad=True))
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        """
        Maps word indices to one-hot vectors, and projects to embedding vectors

        Input:
        x of shape (seq_len, bs)

        Output:
        output of shape (seq_len, bs, embedding_dim)
        """
        ### BEGIN YOUR SOLUTION
        seq_len, bs = x.shape
        x_one_hot = init.one_hot(self.num_embeddings, x, device=self.device, dtype=self.dtype, requires_grad=True) #(seq_len, bs, num_embeddings)
        x_one_hot_reshape = ops.reshape(x_one_hot, (seq_len*bs, self.num_embeddings))
        x_embedding = x_one_hot_reshape @ self.weight # (seq_len*bs, embedding_dim)
        x_embedding_reshape = ops.reshape(x_embedding, (seq_len, bs, self.embedding_dim))
        return x_embedding_reshape
        ### END YOUR SOLUTION