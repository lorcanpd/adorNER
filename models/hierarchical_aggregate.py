import tensorflow as tf


class HierarchicalAggregate(tf.keras.layers.Layer):
    """
    Layer that uses ancestry information from the ontology to share embeddings
    between related ontology concepts.
    """
    def __init__(self, n_concepts, sparse_ancestors, sparse_ancestors_values):
        super(HierarchicalAggregate, self).__init__()
        self.n_concepts = n_concepts
        self.ancestry_sparse_tensor = tf.sparse.reorder(tf.SparseTensor(
            indices=sparse_ancestors,
            values=sparse_ancestors_values,
            dense_shape=[self.n_concepts, self.n_concepts]))

    def build(self, input_shape):
        self.w = self.add_weight(
            'raw_embeddings',
            shape=(self.n_concepts, int(input_shape[-1])),
            initializer=tf.compat.v1.keras.initializers.RandomNormal(0, 0.01),
            trainable=True)
        self.b = self.add_weight(
            'bias',
            shape=(self.n_concepts,),
            initializer=tf.compat.v1.keras.initializers.RandomNormal(0, 0.01),
            trainable=True)

    def call(self, inputs):
        final_w = tf.transpose(
            tf.sparse.sparse_dense_matmul(self.ancestry_sparse_tensor, self.w))
        return tf.matmul(inputs, final_w) + self.b

    def get_global_concept_emb(self):
        return tf.sparse.sparse_dense_matmul(self.ancestry_sparse_tensor,
                                             self.w).numpy()
