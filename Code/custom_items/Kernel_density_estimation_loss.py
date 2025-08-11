import tensorflow as tf


class SimilarityCrossentropy(tf.keras.losses.Loss):
    def __init__(self, batch_size):
        self.reduction = tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE
        self.batch_size = batch_size
        self.name = "similarity_crossentropy"
        super(SimilarityCrossentropy, self).__init__()

    def call(self, y_true, y_pred):
        exponentials = tf.math.exp(tf.clip_by_value(y_pred, clip_value_min=-100000., clip_value_max=75.))
        exponentials_clipped = tf.clip_by_value(exponentials, clip_value_min=1e-7, clip_value_max=100000.)

        log_component = tf.math.divide(
            exponentials_clipped[0],
            tf.math.reduce_sum(exponentials_clipped[1:], axis=0)
        )
        scalar_loss = tf.reduce_mean(-1 * tf.math.log(log_component))

        return tf.fill([self.batch_size], scalar_loss)

    def get_config(self):
        config = super(SimilarityCrossentropy, self).get_config()
        config.update({
            'reduction': self.reduction,
            'batch_size': self.batch_size,
            'name': self.name
        })
        return config

# Simple unit test in eager execution mode
# test_data_samples = tf.random.uniform((42, 556))
# print(SimilarityCrossentropy(batch_size=50)(None, test_data_samples))
