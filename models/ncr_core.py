import tensorflow as tf

from models.hierarchical_aggregate import HierarchicalAggregate
from models.squeeze_excite import ExciteBlock


class NCRCore(tf.keras.Model):

    def __init__(self, config, ont):
        super(NCRCore, self).__init__()
        n_concepts = len(ont.concepts) + 1

        if config.model_type == "sae":
            self.squeeze_excite = True
            self.mean = config.mean
            self.std = config.std
        else:
            self.squeeze_excite = False

        self.num_filters = config.num_filters
        self.concept_dim = config.concept_dim
        self.no_l2norm = config.no_l2norm
        self.flat = config.flat
        self.conv1d = tf.keras.layers.Conv1D(
            self.num_filters,
            1,
            activation=tf.keras.activations.elu,
            kernel_initializer=tf.compat.v1.keras.initializers.RandomNormal(
                0, 0.01
            ),
            bias_initializer=tf.compat.v1.keras.initializers.RandomNormal(
                0, 0.01
            )
        )

        if self.squeeze_excite:
            self.excite = ExciteBlock(config)
            self.scale_filters = tf.keras.layers.Multiply()

        self.max_pool = tf.keras.layers.Lambda(
            lambda z: tf.keras.backend.max(z, axis=1)
        )
        self.dense1 = tf.keras.layers.Dense(
            self.concept_dim,
            activation=tf.keras.activations.relu,
            kernel_initializer=tf.compat.v1.keras.initializers.RandomNormal(
                0, 0.1
            ),
            bias_initializer=tf.compat.v1.keras.initializers.RandomNormal(
                0, 0.01
            )
        )

        if not self.no_l2norm:
            self.l2_norm = tf.keras.layers.Lambda(
                lambda z: tf.keras.backend.l2_normalize(z, axis=1))

        if self.flat:
            self.dense2 = tf.keras.layers.Dense(
                n_concepts,
                kernel_initializer=tf.compat.v1.keras.initializers.RandomNormal(
                    0, 0.01
                )
            )
        else:
            self.hi_ag = HierarchicalAggregate(
                n_concepts, ont.sparse_ancestors, ont.sparse_ancestors_values)

    def call(self, inputs):
        x = self.conv1d(inputs)

        if self.squeeze_excite:
            # Calc. means and SDs for non-zero elements of the feature maps.
            non_zero = tf.cast(tf.not_equal(x, tf.zeros_like(x)), tf.float32)
            sum = tf.math.reduce_sum(x, axis=-2)
            num = tf.math.reduce_sum(non_zero, axis=-2)
            mu = tf.divide(sum, num)
            std = tf.sqrt(tf.reduce_sum(
                ((x - tf.expand_dims(mu, axis=-2)) * non_zero) ** 2,
                axis=-2) / num)

            if self.mean and self.std:
                x2 = tf.concat([mu, std], axis=-1)
            elif self.std and not self.mean:
                x2 = std
            elif self.mean and not self.std:
                x2 = mu

            # if tf.experimental.numpy.any(tf.math.is_nan(x2)) or tf.experimental.numpy.any(tf.math.is_inf(x2)):
            #     breakpoint()
            # TODO: Need to ensure 0 length sequences don't make it from the
            #  ontology.
            x2 = tf.where(tf.math.is_nan(x2), tf.zeros_like(x2), x2)
            x2 = tf.where(tf.math.is_inf(x2), tf.zeros_like(x2), x2)

            x2 = self.excite(x2)

            # if tf.experimental.numpy.any(tf.math.is_nan(x2)) or tf.experimental.numpy.any(tf.math.is_inf(x2)):
            #     breakpoint()

        x = self.max_pool(x)

        if self.squeeze_excite:
            x = self.scale_filters([x, x2])

        x = self.dense1(x)

        if not self.no_l2norm:
            x = self.l2_norm(x)

        if self.flat:
            x = self.dense2(x)
        else:
            x = self.hi_ag(x)

        return x


# import tensorflow as tf
#
# import numpy as np
#
# a = tf.convert_to_tensor([[1.0, 2.0, 3.0], [np.nan, np.inf, 5.0]], tf.float32)
#
# tf.clip_by_value(a, -1e12, 1e12)
#
# tf.where(tf.math.is_nan(a), tf.zeros_like(a), a)
#
# tf.where(tf.math.is_inf(a), tf.zeros_like(a), a)
