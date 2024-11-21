from typing import List
from needle.autograd import Tensor
import needle.backend_ndarray.ndarray as ndarray
from needle import ops
import needle.init as init
import numpy as np
from .nn_sequence import Embedding
from .nn_basic import (
    Parameter, 
    Module, 
    ReLU,
    Dropout,
    LayerNorm1d,
    Linear,
    Sequential,
    Residual
)


class MultiHeadAttention(Module):
    """
    The multi-head self attention module.
    """
    def __init__(
        self,
        *,
        dropout = 0.,
        causal = False,
        device = None,
        dtype = "float32",
    ):

        super().__init__()

        self.device = device
        self.dtype = dtype

        self.causal = causal
        self.dropout = Dropout(dropout)

    def create_causal_mask(self, i, j, device):
        """
        return a triangular causal mask.
        """
        mask = -np.finfo(np.float32).max * np.triu(
            np.ones((1, 1, i, j), dtype=np.float32), j - i + 1)

        return ndarray.array(
            mask, device=device)

    def matmul(self, a, b_transpose):
        """
        batched matrix multiplication;
        """
        a_shape = (*a.shape[:-1], 1, *a.shape[-1:])
        a = a.reshape(a_shape)

        b_transpose_shape = (*b_transpose.shape[:-2], 1, *b_transpose.shape[-2:])
        b_transpose = b_transpose.reshape(b_transpose_shape)

        broadcast_shape = list(a_shape)
        broadcast_shape[-2] = b_transpose_shape[-2]
        a = a.broadcast_to(broadcast_shape)

        broadcast_shape = list(b_transpose_shape)
        broadcast_shape[-3] = a_shape[-3]
        b_transpose = b_transpose.broadcast_to(broadcast_shape)

        return (a * b_transpose).sum(len(a.shape) - 1)

    def softmax(self, logit):
        """
        The softmax function; 
        """
        max_val = Tensor(
            logit.realize_cached_data().max(axis=3),
            device=logit.device,
            dtype=logit.dtype,
            requires_grad=False
        )

        max_val = max_val.reshape((*logit.shape[:-1], 1))
        max_val = max_val.broadcast_to(logit.shape)

        probs = ops.exp(logit - max_val)

        denom = probs.sum(axes=3)
        denom = denom.reshape((*logit.shape[:-1], 1))
        denom = denom.broadcast_to(logit.shape)

        return probs / denom

    def forward(
        self,
        q, k, v,
    ):
        """
        The forward function of the MultiHeadAttention activation function.
        Input: three states q, k, v, with shape (batch_size, num_head, seq_len, dim_head)
        Output: the activation output `result` and attention softmax probability `probs` (with dropout applied)
        """
        batch_size, num_head, queries_len, q_dim = q.shape
        _, _, keys_values_len, k_dim = k.shape
        _, _, _, v_dim = v.shape

        assert q_dim == k_dim == v_dim

        result = None
        probs = None

        ### BEGIN YOUR SOLUTION
        # Compute scaled dot-product attention scores
        attn_scores = self.matmul(q, k) / np.sqrt(q_dim)
    
        # Apply causal masking if needed
        if self.causal:
            attn_scores_mask = self.create_causal_mask(queries_len, keys_values_len, device=self.device)
            attn_scores_mask = attn_scores_mask.broadcast_to(attn_scores.shape)
            attn = attn_scores + attn_scores_mask
        else:
            attn = attn_scores
    
        # Apply softmax and dropout to get the attention probabilities
        probs = self.softmax(attn)
        probs = self.dropout(probs)
    
        # Compute the final output
        result = self.matmul(probs, v.transpose((2, 3)))
        ### END YOUR SOLUTION

        return result, probs


class AttentionLayer(Module):

    def __init__(
        self,
        q_features: int,
        num_head: int,
        dim_head: int,
        *,
        k_features: int = None,
        v_features: int = None,
        out_features: int = None,
        dropout = 0.,
        causal = True,
        device = None,
        dtype = "float32",
    ):

        super().__init__()

        self.device = device
        self.dtype = dtype

        if k_features is None:
            k_features = q_features
        if v_features is None:
            v_features = q_features
        if out_features is None:
            out_features = q_features

        self.q_features = q_features
        self.k_features = k_features
        self.v_features = v_features
        self.out_features = out_features

        self.num_head = num_head
        self.dim_head = dim_head

        self.prenorm_q = LayerNorm1d(
            q_features, device=device, dtype=dtype)
        self.prenorm_k = LayerNorm1d(
            k_features, device=device, dtype=dtype)
        self.prenorm_v = LayerNorm1d(
            v_features, device=device, dtype=dtype)

        inner_dim = num_head * dim_head
        
        self.q_projection = Linear(
            q_features, inner_dim, bias=False,
            device=device, dtype=dtype)
        self.k_projection = Linear(
            k_features, inner_dim, bias=False,
            device=device, dtype=dtype)
        self.v_projection = Linear(
            v_features, inner_dim, bias=False,
            device=device, dtype=dtype)

        self.attn = MultiHeadAttention(
            dropout=dropout, causal=causal,
            device=device, dtype=dtype)

        self.out_projection = Linear(
            inner_dim, out_features, bias=False,
            device=device, dtype=dtype)

    def forward(
        self,
        q, k=None, v=None,
    ):
        """
        The forward function of the self-attention layer.
        Input: `q` with shape (batch_size, q_len, q_dim)
               `k` (if not None) with shape (batch_size, kv_len, k_dim)
               `v` (if not None) with shape (batch_size, kv_len, v_dim)
        Output: the output `result` with shape (batch_size, kv_len, out_features)
        """

        if k is None:
            k = q
        if v is None:
            v = q

        batch_size, queries_len, q_dim = q.shape
        _, keys_values_len, k_dim = k.shape
        _, _, v_dim = v.shape

        result = None

        ### BEGIN YOUR SOLUTION
        pren_q = self.prenorm_q(q.reshape((batch_size*queries_len, q_dim))) # (B*T, D')
        pren_k = self.prenorm_k(k.reshape((batch_size*keys_values_len, k_dim))) # (B*T, D')
        pren_v = self.prenorm_v(v.reshape((batch_size*keys_values_len, v_dim))) # (B*T, D')

        q_pron = self.q_projection(pren_q) #(B*T, num_head*dim_head)
        k_pron = self.k_projection(pren_k) #(B*T, num_head*dim_head)
        v_pron = self.v_projection(pren_v) #(B*T, num_head*dim_head)

        Q = q_pron.reshape((batch_size, queries_len, self.num_head, self.dim_head)).transpose((1, 2)) #(B, H, T, D)
        K = k_pron.reshape((batch_size, keys_values_len, self.num_head, self.dim_head)).transpose((1, 2)) #(B, H, T, D)
        V = v_pron.reshape((batch_size, keys_values_len, self.num_head, self.dim_head)).transpose((1, 2)) #(B, H, T, D)

        X, prob = self.attn(Q, K, V) # (B, H, T, D)
        X_trans = X.transpose((1, 2)) # (B, T, H, D)
        X_reshape = X_trans.reshape((X_trans.shape[0]*X_trans.shape[1], self.num_head*self.dim_head)) # ((B*T), (H*D))

        X_pron = self.out_projection(X_reshape)
        result = X_pron.reshape((X_trans.shape[0], X_trans.shape[1], X_pron.shape[1])) #(B, T, out_features)
        ### END YOUR SOLUTION

        return result


class TransformerLayer(Module):

    def __init__(
        self,
        q_features: int,
        num_head: int,
        dim_head: int,
        hidden_size: int,
        *,
        dropout = 0.,
        causal = True,
        device = None,
        dtype = "float32",
    ):

        super().__init__()

        self.device = device
        self.dtype = dtype
        self.q_features = q_features
        self.num_head = num_head
        self.dim_head = dim_head
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.causal = causal

        ### BEGIN YOUR SOLUTION
        '''
        self.residblock1 = self.ResidualBlock1(q_features, num_head, dim_head, hidden_size, dropout, causal, device, dtype)
        self.residblock2 = self.ResidualBlock2(q_features, num_head, dim_head, hidden_size, dropout, causal, device, dtype)
        '''
        self.attention_layer = AttentionLayer(
            q_features=q_features,
            num_head=num_head,
            dim_head=dim_head,
            k_features=None,
            v_features=None,
            out_features=None,
            dropout=dropout,
            causal=causal,
            device=device,
            dtype=dtype
        )

        self.layernorm     = LayerNorm1d(q_features, device=device, dtype=dtype)
        self.linear1_layer = Linear(q_features, hidden_size, bias=True, device=device, dtype=dtype)
        self.linear2_layer = Linear(hidden_size, q_features, bias=True, device=device, dtype=dtype)
        self.relu1         = ReLU()
        self.dropout1      = Dropout(p=dropout)
        self.dropout2      = Dropout(p=dropout)
        self.dropout3      = Dropout(p=dropout)
        ### END YOUR SOLUTION

    def ResidualBlock1(self, q_features: int, num_head: int, dim_head: int, hidden_size: int, dropout = 0., causal = True, device = None, dtype = "float32"):
        out_module = Residual(
            Sequential(
                AttentionLayer(
                    q_features=q_features,
                    num_head=num_head,
                    dim_head=dim_head,
                    k_features=None,
                    v_features=None,
                    out_features=None,
                    dropout=dropout,
                    causal=causal,
                    device=device,
                    dtype=dtype
                ),
                Dropout(p=dropout),
            )
        )

        return out_module

    def ResidualBlock2(self, q_features: int, num_head: int, dim_head: int, hidden_size: int, dropout = 0., causal = True, device = None, dtype = "float32"):
        out_module = Residual(
            Sequential(
                LayerNorm1d(q_features, device=device, dtype=dtype),
                Linear(q_features, hidden_size, bias=True, device=device, dtype=dtype),
                ReLU(),
                Dropout(p=dropout),
                Linear(hidden_size, q_features, bias=True, device=device, dtype=dtype),
                Dropout(p=dropout)
            )
        )
        return out_module

    def forward(
        self,
        x
    ):
        """
        The forward function of a Transformer Layer.
        Input: the hidden states from previous layers `x` with shape (batch_size, seq_len, x_dim)
        Ouput: the hidden states after the Transformer Layer `x` with shape (batch_size, seq_len, x_dim)
        """

        batch_size, seq_len, x_dim = x.shape

        ### BEGIN YOUR SOLUTION
        '''
        x = self.residblock1(x)
        x = x.reshape((batch_size*seq_len, x_dim))
        x = self.residblock2(x)
        x = x.reshape((batch_size, seq_len, x_dim))
        '''
        orig_x1 = x
        x = self.attention_layer(x) #(B, T, q_features)
        x = orig_x1 + self.dropout1(x) #(B, T, q_features)
        orig_x2 = x
        x = self.layernorm(x.reshape((batch_size*seq_len, x_dim))) #(B*T, q_features)
        x = self.linear1_layer(x) #(B*T, hidden_size)
        x = self.relu1(x) #(B*T, hidden_size)
        x = self.dropout2(x) #(B*T, hidden_size)
        x = self.linear2_layer(x) #(B*T, q_features)
        x = self.dropout3(x) #(B*T, q_features)
        x = x.reshape((batch_size, seq_len, x_dim)) #(B, T, q_features)
        x = x + orig_x2
        ### END YOUR SOLUTION

        return x


class Transformer(Module):

    def __init__(
        self,
        embedding_size: int,
        hidden_size: int,
        num_layers: int, 
        *,
        num_head: int = 8,
        dim_head: int = 32,
        dropout = 0.,
        causal = True,
        device = None,
        dtype = "float32",
        batch_first = False,
        sequence_len = 2048
    ):

        super().__init__()

        self.device = device
        self.dtype = dtype
        self.batch_first = batch_first

        ### BEGIN YOUR SOLUTION
        self.embedding_layer = Embedding(num_embeddings=sequence_len, embedding_dim=embedding_size, device=device, dtype=dtype)
        self.layers = []
        for _ in range(num_layers):
            self.layers.append(
                TransformerLayer(
                    q_features=embedding_size,
                    num_head=num_head,
                    dim_head=dim_head,
                    hidden_size=hidden_size,
                    dropout=dropout,
                    causal=causal,
                    device=device,
                    dtype=dtype
                )
            )        

        ### END YOUR SOLUTION

    def forward(
        self,
        x, h=None
    ):

        if not self.batch_first:
            x = ops.transpose(x, axes=(0, 1))

        ### BEGIN YOUR SOLUTION
        batch_size, seq_len, input_dim = x.shape
        positions = np.broadcast_to(np.arange(seq_len, dtype=x.dtype).reshape((seq_len, 1)), (seq_len, batch_size)) #(seq_len, batch_size)
        positions_tensor = Tensor(positions, device=self.device, dtype=self.dtype, requires_grad=False)
        positional_encodings = self.embedding_layer(positions_tensor) #(seq_len, batch_size, input_dim)
        positional_encodings = positional_encodings.transpose((0, 1)) #(batch_size, seq_len, input_dim)

        x = x + positional_encodings
        for layer in self.layers:
            x = layer(x)
        ### END YOUR SOLUTION

        if not self.batch_first:
            x = ops.transpose(x, axes=(0, 1))

        return x, init.zeros_like(x)