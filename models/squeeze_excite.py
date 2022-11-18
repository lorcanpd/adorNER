import tensorflow as tf


class ExciteBlock(tf.keras.layers.Layer):
    """
    A layer that learns to perform a series of transformations on a vector input
    (typically a vector containing the averages of a number of feature maps) in
    order to extract useful signals to scale/weight the original feature maps.
    """

    def __init__(self, config):
        super(ExciteBlock, self).__init__()
        if config.mean and config.std:
            self.double = True
        else:
            self.double = False

        self.ratio = config.compression_ratio

    def build(self, input_shape):

        if self.double:
            compress_dim = input_shape[-1] // (self.ratio * 2)
            uncompress_dim = input_shape[-1] // 2
        else:
            compress_dim = input_shape[-1] // self.ratio
            uncompress_dim = input_shape[-1]

        ew1_init = tf.keras.initializers.he_normal()
        eb1_init = tf.keras.initializers.he_normal()
        self.ew1 = tf.Variable(
            initial_value=ew1_init(shape=(compress_dim, input_shape[-1]),
                                   dtype=tf.float32),
            trainable=True,
            name='excite_w1'
        )
        self.eb1 = tf.Variable(
            initial_value=eb1_init(shape=(compress_dim,),
                                   dtype=tf.float32),
            trainable=True,
            name='excite_b1'
        )
        ew2_init = tf.keras.initializers.he_normal()
        eb2_init = tf.keras.initializers.he_normal()
        self.ew2 = tf.Variable(
            initial_value=ew2_init(shape=(uncompress_dim, compress_dim),
                                   dtype=tf.float32),
            trainable=True,
            name='excite_w2'
        )
        self.eb2 = tf.Variable(
            initial_value=eb2_init(shape=(uncompress_dim,),
                                   dtype=tf.float32),
            trainable=True,
            name='excite_b2'
        )

    def call(self, inputs):
        x = tf.linalg.matvec(self.ew1, inputs) + self.eb1
        x = tf.nn.relu(x)
        x = tf.linalg.matvec(self.ew2, x) + self.eb2

        return tf.math.sigmoid(x)

