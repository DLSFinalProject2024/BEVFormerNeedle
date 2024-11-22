"""The module.
"""
from typing import List
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np
from .nn_basic import Parameter, Module


class Sigmoid(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return (1 + ops.exp(-x)) ** (-1)  # Not supported 1 / (1 + ops.exp(-x))

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
        self.device = device
        self.dtype = dtype

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias # bool
        k = 1 / hidden_size ** 0.5
        self.W_ih = Parameter(init.rand(input_size, hidden_size, low=-k, high=k, device=device, dtype=dtype))
        self.W_hh = Parameter(init.rand(hidden_size, hidden_size, low=-k, high=k, device=device, dtype=dtype))
        self.bias_ih = Parameter(init.rand(hidden_size, low=-k, high=k, device=device, dtype=dtype)) if bias else None
        self.bias_hh = Parameter(init.rand(hidden_size, low=-k, high=k, device=device, dtype=dtype)) if bias else None
        self.nonlinearity = ops.tanh if nonlinearity == 'tanh' else ops.relu


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
        bs, input_size = X.shape
        if h is None:
            h = init.zeros(bs, self.hidden_size, device=self.device, dtype=self.dtype)
        h_next = X @ self.W_ih + h @ self.W_hh # shape: (bs, hidden_size)
        if self.bias:
            h_next += self.bias_ih.reshape((1, self.hidden_size)).broadcast_to(h_next.shape)
            h_next += self.bias_hh.reshape((1, self.hidden_size)).broadcast_to(h_next.shape)

        return self.nonlinearity(h_next)


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
        self.device = device
        self.dtype = dtype

        self.num_layers = num_layers
        self.hidden_size = hidden_size

        self.rnn_cells = [RNNCell(input_size, hidden_size, bias, nonlinearity, device, dtype)]
        for _ in range(1, num_layers):
            self.rnn_cells.append(
                RNNCell(hidden_size, hidden_size, bias, nonlinearity, device, dtype)
            )

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
        seq_len, bs, input_size = X.shape
        h0_shape = (self.num_layers, bs, self.hidden_size)
        if h0 is None:
            h0 = init.zeros(*h0_shape, device=self.device, dtype=self.dtype)

        outputs = [] # (seq_len, bs, hidden_size)
        X_list = list(ops.split(X, axis=0)) # seq_len * (bs, input_size)
        h_prev = list(ops.split(h0, axis=0)) # num_layers * (bs, hidden_size)
        for x_t in X_list:
            for i, rnn_cell in enumerate(self.rnn_cells):
                h_prev[i] = rnn_cell.forward(x_t, h_prev[i])
                x_t = h_prev[i]

            outputs.append(h_prev[-1])
        
        return ops.stack(outputs, axis=0), ops.stack(h_prev, axis=0)


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
        self.device = device
        self.dtype = dtype

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias # bool
        k = 1 / hidden_size ** 0.5
        self.W_ih = Parameter(init.rand(input_size, 4 * hidden_size, low=-k, high=k, device=device, dtype=dtype))
        self.W_hh = Parameter(init.rand(hidden_size, 4 * hidden_size, low=-k, high=k, device=device, dtype=dtype))
        self.bias_ih = Parameter(init.rand(4 * hidden_size, low=-k, high=k, device=device, dtype=dtype)) if bias else None
        self.bias_hh = Parameter(init.rand(4 * hidden_size, low=-k, high=k, device=device, dtype=dtype)) if bias else None

        self.tanh = ops.Tanh()
        self.sigmoid = Sigmoid()

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
        bs, input_size = X.shape
        # breakpoint()
        if h is None:
            h = init.zeros(bs, self.hidden_size, device=self.device, dtype=self.dtype)
            c = init.zeros(bs, self.hidden_size, device=self.device, dtype=self.dtype)
        else:
            h, c = h

        gates = X @ self.W_ih + h @ self.W_hh  # shape: (bs, 4*hidden_size)
        if self.bias:
            gates += self.bias_ih.reshape((1, 4 * self.hidden_size)).broadcast_to(gates.shape)
            gates += self.bias_hh.reshape((1, 4 * self.hidden_size)).broadcast_to(gates.shape)
        
        _gates = list(ops.split(gates, axis=1))  # 4 * hidden_size * (bs, )

        # Split all gates and them stack each by hidden_size
        gates = []  # 4 * (bs, hidden_size)
        for i in range(4):
            gate = _gates[i * self.hidden_size : (i+1) * self.hidden_size]
            gates.append(ops.stack(gate, axis=1))
        
        i = self.sigmoid(gates[0])
        f = self.sigmoid(gates[1])
        g = self.tanh(gates[2])
        o = self.sigmoid(gates[3])

        c_next = f * c + i * g
        h_next = o * self.tanh(c_next)
        return h_next, c_next


class LSTM(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, device=None, dtype="float32"):
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
        super().__init__()
        self.device = device
        self.dtype = dtype

        self.num_layers = num_layers
        self.hidden_size = hidden_size

        self.lstm_cells = [LSTMCell(input_size, hidden_size, bias, device, dtype)]
        for _ in range(1, num_layers):
            self.lstm_cells.append(
                LSTMCell(hidden_size, hidden_size, bias, device, dtype)
            )

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
            c_n of shape (num_layers, bs, hidden_size) containing the final hidden cell state for each element in the batch.
        """
        seq_len, bs, input_size = X.shape
        h0c0_shape = (self.num_layers, bs, self.hidden_size)
        if h is None:
            h0 = init.zeros(*h0c0_shape, device=self.device, dtype=self.dtype)
            c0 = init.zeros(*h0c0_shape, device=self.device, dtype=self.dtype)
        else:
            h0, c0 = h

        X_list = list(ops.split(X, axis=0)) # seq_len * (bs, input_size)
        h_prev = list(ops.split(h0, axis=0)) # num_layers * (bs, hidden_size)
        c_prev = list(ops.split(c0, axis=0)) # num_layers * (bs, hidden_size)
        output = [] # (seq_len, bs, hidden_size)
        for x_t in X_list:
            for i, lstm_cell in enumerate(self.lstm_cells):
                h_prev[i], c_prev[i] = lstm_cell.forward(x_t, (h_prev[i], c_prev[i]))
                x_t = h_prev[i]

            output.append(h_prev[-1])
    
        output = ops.stack(output, axis=0)
        h_prev = ops.stack(h_prev, axis=0)
        c_prev = ops.stack(c_prev, axis=0)
        return output, (h_prev, c_prev)

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
        self.device = device
        self.dtype = dtype
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        self.weight = Parameter(init.randn(num_embeddings, embedding_dim, device=device, dtype=dtype))

    def forward(self, x: Tensor) -> Tensor:
        """
        Maps word indices to one-hot vectors, and projects to embedding vectors

        Input:
        x of shape (seq_len, bs)

        Output:
        output of shape (seq_len, bs, embedding_dim)
        """
        one_hot = init.one_hot(self.num_embeddings, x, device=self.device, dtype=self.dtype)  # shape: (seq_len, bs, num_embeddings)
        seq_len, bs = one_hot.shape[:2]
        # Note: now only support 2x2 matrix multiplication
        one_hot = one_hot.reshape((seq_len * bs, self.num_embeddings))
        ret = one_hot @ self.weight
        return ret.reshape((seq_len, bs, self.embedding_dim))