import random

import tensorflow as tf
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import random_ops

from models.hierarchical_aggregate import HierarchicalAggregate
from utils.activations import gelu


class AttentionHead(tf.keras.layers.Layer):
    """
    A scaled attention head.
    ...
    Attributes
    ----------
    scale : int
        the size of the scaling window.
    num_heads : int
        the total number of attention heads.

    """
    def __init__(self, scale, num_heads=1):
        super(AttentionHead, self).__init__()
        self.h = num_heads
        self.scale = scale

    def build(self, input_shape):
        qkv_dim = input_shape[-1] // self.h
        self.sqrt_dim = tf.sqrt(tf.cast(qkv_dim, tf.float32))
        head_id = "%0.9d" % random.randint(0, 999999999)
        init = tf.keras.initializers.he_uniform()
        self.wqw = tf.Variable(
            initial_value=init(shape=(input_shape[-1], qkv_dim),
                               dtype=tf.float32),
            trainable=True,
            name=f'wq_w_{head_id}'
        )
        self.wkw = tf.Variable(
            initial_value=init(shape=(input_shape[-1], qkv_dim),
                               dtype=tf.float32),
            trainable=True,
            name=f'wk_w_{head_id}'
        )
        self.wvw = tf.Variable(
            initial_value=init(shape=(input_shape[-1], qkv_dim),
                               dtype=tf.float32),
            trainable=True,
            name=f'wv_w_{head_id}'
        )

    def context_fn(self, matrix, idx, scale):
        """
        This function extracts the context of a given index of a tensor, within
        a specified number of indices of the index. If the context window
        exceeds the dimensions of the input matrix, the output is padded with
        zeros to ensure a consistent output.
        ...
        Attributes
        ----------
        matrix : n-D `Tensor` of type `float32`
            matrix tensor from which to extract a subset.
        idx : int
            index from around which the context will be extracted
        scale : int
            the size of the context window

        returns : n-D tensor
            a subset of the input `matrix` corresponding to the context of `idx`
            within `scale` indecies of `idx`
        """
        if scale == 0:
            columns = [idx]
        else:
            columns = [x for x in range(idx-scale, idx+scale)
                       if x >= 0 and x < matrix.shape[-2]]

        context = tf.gather(matrix, columns, axis=-2)
        zero_cols = [-1*x if x<0 else x-matrix.shape[-2]
                     for x in range(idx-scale, idx+scale)
                     if x < 0 or x >= matrix.shape[-2]]

        if idx-scale < 0:
            shape = tf.gather(matrix, zero_cols, axis=-2)
            context = tf.concat([tf.zeros_like(shape), context], axis=-2)
        elif idx+scale > matrix.shape[-2]:
            shape = tf.gather(matrix, zero_cols, axis=-2)
            context = tf.concat([context, tf.zeros_like(shape)], axis=-2)

        return context

    def call(self, inputs, training=False, **kwargs):
        query = tf.matmul(inputs, self.wqw)
        key = tf.matmul(inputs, self.wkw)
        value = tf.matmul(inputs, self.wvw)

        output = []
        for j in range(inputs.shape[1]):
            q_j = self.context_fn(query, j, 0)
            k_j = self.context_fn(key, j, self.scale)
            v_j = self.context_fn(value, j, self.scale)

            if training:
                q_j = tf.nn.dropout(q_j, rate=0.1)  # Feature dropout 3.
                k_j = tf.nn.dropout(k_j, rate=0.1)
                v_j = tf.nn.dropout(v_j, rate=0.1)

            a = tf.matmul(q_j, k_j, transpose_b=True)

            if training:
                a = tf.nn.dropout(a, rate=0.1)  # Feature dropout 1.

            score = tf.divide(a, self.sqrt_dim)
            alignment = tf.nn.softmax(score, axis=2)
            context = tf.matmul(alignment, v_j)
            output.append(context)

        return tf.concat(output, axis=-2)

@tf.function
def drop_head(head_list, rate=0.25):
    """
    Function for applying dropout to entire attention heads.
    ...
    Attributes
    ----------
    head_list : list
        a list containing the attention heads
    rate : float
        the probability of each attention head being dropped

    returns : list
        a list containing the attention heads with the dropout applied
    """
    return [head/(1-rate) if random.random() > rate else head*0
            for head in head_list]


class MultiScaleSelfAttention(tf.keras.Model):
    """
    A Multi-Scale Self-Attention block.
    ...
    Attributes
    ----------
    input_size :
        the length of input sequence
    num_heads : int
        the number of attention heads
    scales : list(int)
        a list containing the scale parameters for each attention head

    """

    def __init__(self, input_size, num_heads, scales):
        super(MultiScaleSelfAttention, self).__init__()
        self.h = num_heads
        head_scales = (scales * (num_heads // len(scales) + 1))[:num_heads]
        self.heads = [AttentionHead(scale, self.h) for scale in head_scales]
        self.wo = tf.keras.layers.Dense(input_size)
        self.norm1 = tf.keras.layers.LayerNormalization(axis=-1)
        self.norm2 = tf.keras.layers.LayerNormalization(axis=-1)
        self.ff1 = tf.keras.layers.Dense(
            input_size*2,
            activation=gelu,
            kernel_initializer=tf.keras.initializers.he_uniform(),
            bias_initializer=tf.keras.initializers.he_uniform()
        )
        self.ff2 = tf.keras.layers.Dense(
            input_size,
            activation=tf.keras.activations.linear,
            kernel_initializer=tf.keras.initializers.he_uniform(),
            bias_initializer=tf.keras.initializers.he_uniform()
        )

    def call(self, inputs, training=False):
        # Post-Layer-Normalisation Transformer set-up. Requires warm-up stage.
        if training:
            heads = drop_head([head(inputs, training) for head in self.heads])
        else:
            heads = [head(inputs) for head in self.heads]

        z = tf.concat(heads, axis=2)
        sa = self.wo(z)
        act = tf.add(inputs, sa)
        norm1 = self.norm1(act)
        ff1 = self.ff1(norm1)
        if training:
            ff1 = tf.nn.dropout(ff1, rate=0.1)  # Feature dropout 2.
        ff2 = self.ff2(ff1)
        out = tf.add(act, ff2)
        out = self.norm2(out)

        return out


class PNMultiScaleSelfAttention(tf.keras.Model):
    """
    A pre-normalisation residual configuration of the  Multi-Scale
    Self-Attention block.
    ...
    Attributes
    ----------
    input_size :
        the length of input sequence
    num_heads : int
        the number of attention heads
    scales : list(int)
        a list containing the scale parameters for each attention head

    """

    def __init__(self, input_size, num_heads, scales):
        super(PNMultiScaleSelfAttention, self).__init__()
        self.h = num_heads
        head_scales = (scales * (num_heads // len(scales) + 1))[:num_heads]
        self.heads = [AttentionHead(scale, self.h) for scale in head_scales]
        self.wo = tf.keras.layers.Dense(input_size)
        self.norm1 = tf.keras.layers.LayerNormalization(axis=-1)
        self.norm2 = tf.keras.layers.LayerNormalization(axis=-1)
        self.ff1 = tf.keras.layers.Dense(
            input_size * 2,
            activation=gelu,
            kernel_initializer=tf.keras.initializers.he_uniform(),
            bias_initializer=tf.keras.initializers.he_uniform()
        )
        self.ff2 = tf.keras.layers.Dense(
            input_size,
            activation=tf.keras.activations.linear,
            kernel_initializer=tf.keras.initializers.he_uniform(),
            bias_initializer=tf.keras.initializers.he_uniform()
        )

    def call(self, inputs, training=False):
        # Pre-Layer-Normalistation Transformer set-up. Doesn't need warm-up
        # stage.
        norm1 = self.norm1(inputs)
        if training:
            heads = drop_head(
                [head(norm1, training) for head in self.heads])
        else:
            heads = [head(norm1) for head in self.heads]
        z = tf.concat(heads, axis=2)
        sa = self.wo(z)
        act = tf.add(inputs, sa)
        norm2 = self.norm2(act)
        ff1 = self.ff1(norm2)
        if training:
            ff1 = tf.nn.dropout(ff1, rate=0.1)  # Feature dropout 2.
        ff2 = self.ff2(ff1)
        out = tf.add(act, ff2)

        return out


@tf.function
def drop_input(inputs, lens, rate=0.1):
    """
    A function that applies a dropout to the input sequence of word embeddings.
    Entire embeddings are randomly dropped out as opposed to random elements
    across all embeddings. Input sequences of fewer than two tokens are not
    considered for dropout.
    ...
    Attributes
    ----------
    inputs : n-D (n>1) `Tensor` of type `float32`
        a sequence, or batch of sequences, of word embeddings
    lens :
        the actual length of each sequence - not including the padding tokens
    rate :
        probability of a given embedding in a sequence being dropped

    returns : n-D `Tensor` of type `float32`
        a sequence, or batch of sequences, with dropout applied to the
        embeddings
    """
    keep_mask_a = tf.expand_dims(lens < 2, -1)
    shape_a = tf.shape(inputs)[0]
    shape_b = tf.shape(inputs)[1]
    keep_mask_b = random_ops.random_uniform(shape=(shape_a, shape_b),
                                            dtype=tf.float32) >= rate
    keep_mask = tf.expand_dims(
        tf.cast(
            tf.math.logical_or(keep_mask_a, keep_mask_b),
            tf.float32),
        -1)
    return gen_math_ops.mul(inputs, keep_mask) / (1-rate)


class MSSACore(tf.keras.Model):
    """
    Multiple Multi-Scale Self-Attention blocks stacked together, leading into an
    element-wise averaging layer and then into a Feed-Forward Network.
    ...
    Attributes
    ----------
    config :
        blah
    ont :
        blah
    scales :
        the combinations of different scaling parameters to be used by the
        attention heads in each block
    """

    def __init__(self, config, ont):
        super(MSSACore, self).__init__()

        try:
            n_concepts = ont.n_concepts + 1
        except AttributeError:
            n_concepts = len(ont.concepts) + 1

        self.flat = config.flat
        self.no_l2norm = config.no_l2norm
        self.input_vec_size = 512 #if config.word_emb else 1024
        self.pre_norm = config.pre_norm
        self.concept_dim = config.concept_dim
        self.scales = config.scales
        self.num_heads = config.num_heads

        if not self.pre_norm:
            self.mssa = [MultiScaleSelfAttention(input_size=self.input_vec_size,
                                                 scales=self.scales[i],
                                                 num_heads=self.num_heads)
                         for i in range(len(self.scales))] # Infer num_blocks from scales
        else:
            self.mssa = [PNMultiScaleSelfAttention(input_size=self.input_vec_size,
                                                   scales=self.scales[i],
                                                   num_heads=self.num_heads)
                         for i in range(len(self.scales))]

        self.FF1 = tf.keras.layers.Dense(
            self.input_vec_size*2,
            activation=gelu,
            kernel_initializer=tf.keras.initializers.he_uniform(),
            bias_initializer=tf.keras.initializers.he_uniform()
        )
        self.FF2 = tf.keras.layers.Dense(
            self.concept_dim,
            activation=gelu,
            kernel_initializer=tf.keras.initializers.he_uniform(),
            bias_initializer=tf.keras.initializers.he_uniform()
        )

        if not self.no_l2norm:
            self.l2_norm = tf.keras.layers.Lambda(
                lambda z: tf.keras.backend.l2_normalize(z, axis=1))

        if self.flat:
            self.dense3 = tf.keras.layers.Dense(
                n_concepts,
                kernel_initializer=tf.compat.v1.keras.initializers.RandomNormal(
                    0, 0.01
                )
            )
        else:
            self.hi_ag = HierarchicalAggregate(
                n_concepts, ont.sparse_ancestors, ont.sparse_ancestors_values)

    def call(self, inputs, lens, training=False):

        block_input = inputs

        if training and random.random() > 0.5:
            block_input = drop_input(block_input, lens, rate=0.2)

        for mssa_block in self.mssa:
            block_input = mssa_block(block_input, training=training)

        x = block_input

        x = tf.math.divide(tf.math.reduce_sum(x, axis=1),
                           tf.sqrt(tf.expand_dims(tf.cast(lens, tf.float32),
                                                  -1)))
        x = self.FF1(x)

        if training:
            x = tf.nn.dropout(x, 0.1)  # Own dropout addition.

        if not self.no_l2norm:
            x = self.l2_norm(x)

        x = self.FF2(x)

        if not self.no_l2norm:
            x = self.l2_norm(x)

        if self.flat:
            x = self.dense3(x)
        else:
            x = self.hi_ag(x)

        return x

