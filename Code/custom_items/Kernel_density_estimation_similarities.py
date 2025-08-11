import tensorflow as tf


class BilinearSimilarity(tf.keras.layers.Layer):
    def __init__(self, initializer, kernel_regularizer):
        self.initializer = tf.keras.initializers.get(initializer)
        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        super(BilinearSimilarity, self).__init__()

    def build(self, input_shape):
        self.kernel = self.add_weight(shape=(input_shape[-2], input_shape[-1], input_shape[-1]),
                                      initializer=self.initializer,
                                      regularizer=self.kernel_regularizer, trainable=True, name="kernel")
        super(BilinearSimilarity, self).build(input_shape)

    def call(self, inputs, **kwargs):
        other_pdfs = tf.expand_dims(inputs[1:], axis=-1)
        current_pdfs_stack = tf.expand_dims(tf.stack([inputs[0]] * other_pdfs.shape[0]), axis=-2)
        kernel_stack = tf.stack([self.kernel] * other_pdfs.shape[0])
        return tf.squeeze(tf.matmul(tf.matmul(current_pdfs_stack, kernel_stack), other_pdfs))

    def get_config(self):
        config = super(BilinearSimilarity, self).get_config()
        config.update({
            'initializer': tf.keras.initializers.serialize(self.initializer),
            'kernel_regularizer': tf.keras.regularizers.serialize(self.kernel_regularizer)
        })
        return config


# To be used in combination with a Keras Lambda layer
def cosine_similarity(inputs):
    func = lambda x: tf.math.divide(
        tf.tensordot(x[0], x[1], 1),
        tf.math.reduce_euclidean_norm(x[0]) * tf.math.reduce_euclidean_norm(x[1])
    )
    return tf.map_fn(fn=func, elems=(tf.stack([inputs[0]] * inputs[1:].shape[0]), inputs[1:]),
                     fn_output_signature=tf.dtypes.float32)

# Simple unit test in eager execution mode
# test_data_samples = tf.random.uniform((42, 556, 100))
# custom_layer = BilinearSimilarity(initializer="random_normal", kernel_regularizer=None)
# # custom_layer2 = tf.keras.layers.Lambda(cosine_similarity)
# output = custom_layer(test_data_samples)
# # output2 = custom_layer2(test_data_samples)
# print(output)
