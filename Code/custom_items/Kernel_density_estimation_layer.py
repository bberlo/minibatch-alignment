import tensorflow as tf
import tensorflow_probability as tfp
from custom_items.utilities import CorrectedSigmoid, CorrectedExp, CorrectedSoftplus, CorrectedReciprocal


# Note: when debugging with tf.debugging.enable_check_numerics use compute_quantiles from custom_items.utilities
# to prevent random NaN const. op placement in Graph mode.


class KernelDensityEstimation(tf.keras.layers.Layer):
    def __init__(self, bank_capacity, bank_initializer, bandwidth, pdf_resolution, quantile_sample_resolution,
                 random_state):
        self.trainable = False
        self.bank_capacity = bank_capacity + 2
        self.bank_initializer = tf.keras.initializers.get(bank_initializer)
        self.bandwidth = bandwidth
        self.pdf_resolution = pdf_resolution
        self.quantile_sample_resolution = quantile_sample_resolution
        self.random_state = random_state
        self.random_generator = tf.random.Generator.from_seed(self.random_state)
        super(KernelDensityEstimation, self).__init__()

    def build(self, input_shape):
        if input_shape[0] is None:
            raise ValueError('KernelDensityEstimation requires a static input layer batch size definition')

        if len(input_shape) > 2:
            if any(x != 1 for x in input_shape[1:-1]):
                raise Exception('KernelDensityEstimation requires a flattened input embedding '
                                '(e.g. with dimensions (Batch size, embedding size) or (Batch size, 1, ..., 1, embedding size))')

        self.negative_example_bank = tf.Variable(
            self.bank_initializer(shape=(self.bank_capacity, input_shape[-1], self.pdf_resolution)),
            trainable=False, name="negative_example_bank")
        super(KernelDensityEstimation, self).build(input_shape)

    def call(self, inputs, **kwargs):
        if len(inputs.shape) > 2:
            tf.squeeze(inputs, list(range(0, len(inputs.shape)))[1:-1])

        current_pdfs, transformed_pdfs = tf.map_fn(fn=self.kernel_density_estimator_func, elems=tf.transpose(inputs),
                                                   fn_output_signature=(
                                                       tf.TensorSpec(shape=(self.pdf_resolution,), dtype=tf.float32),
                                                       tf.TensorSpec(shape=(self.pdf_resolution,), dtype=tf.float32))
                                                   )
        previous_pdfs = self.negative_example_bank.sparse_read(0)
        self.negative_example_bank.scatter_nd_update(
            indices=[[0], [1], [tf.random.uniform([], minval=2, maxval=self.bank_capacity - 1, dtype=tf.dtypes.int32)]],
            updates=tf.stack([current_pdfs, transformed_pdfs, previous_pdfs], axis=0))

        return self.negative_example_bank.value()

    def kernel_density_estimator_func(self, batch_sample_features):
        kernel_iterator = lambda layer_input: tfp.distributions.Independent(
            tfp.distributions.Normal(loc=layer_input, scale=self.bandwidth))
        density_estimator = tfp.distributions.MixtureSameFamily(
            mixture_distribution=tfp.distributions.Categorical(
                probs=[1 / batch_sample_features.shape[0]] * batch_sample_features.shape[0]),
            components_distribution=kernel_iterator(batch_sample_features)
        )
        sampled_values = density_estimator.sample(self.quantile_sample_resolution)

        # Reduce min, max ops are more optimal equivalent to taking outer elements from tfp.stats.quantiles() result
        sample_locations = tf.linspace(tf.math.reduce_min(sampled_values), tf.math.reduce_max(sampled_values), self.pdf_resolution)

        # Sample_locations_fix offsets 0 valued sample location to prevent NaN values when used in truediv op
        sample_locations_fix = tf.where(tf.math.not_equal(sample_locations, [0.0]), sample_locations, [1.0e-7])
        current_pdf = density_estimator.prob(sample_locations_fix)

        selection_nr = self.random_generator.uniform(shape=(), minval=0, maxval=6, dtype=tf.int32)
        transformed_pdf = tf.switch_case(branch_index=selection_nr, branch_fns={
            0: lambda: self.transform_sample_dist(distribution=density_estimator,
                                                  bijector=tfp.bijectors.Scale(scale=2.)),
            1: lambda: self.transform_sample_dist(distribution=density_estimator,
                                                  bijector=tfp.bijectors.Shift(shift=-3.)),
            2: lambda: self.transform_sample_dist(distribution=density_estimator, bijector=CorrectedReciprocal()),
            3: lambda: self.transform_sample_dist(distribution=density_estimator, bijector=tfp.bijectors.Chain(
                [tfp.bijectors.Shift(shift=5.), tfp.bijectors.Scale(scale=0.8)])),
            4: lambda: self.transform_sample_dist(distribution=density_estimator, bijector=CorrectedSigmoid()),
            5: lambda: self.transform_sample_dist(distribution=density_estimator, bijector=CorrectedSoftplus()),
            6: lambda: self.transform_sample_dist(distribution=density_estimator, bijector=CorrectedExp()),
        }, default=lambda: self.transform_sample_dist(distribution=density_estimator,
                                                      bijector=tfp.bijectors.Scale(scale=2.)))

        return current_pdf, transformed_pdf

    def transform_sample_dist(self, distribution, bijector):
        transformed_dist = tfp.distributions.TransformedDistribution(
            distribution=distribution,
            bijector=bijector
        )
        transformed_sampled_values = transformed_dist.sample(self.quantile_sample_resolution)

        # Reduce min, max ops are more optimal equivalent to taking outer elements from tfp.stats.quantiles() result
        sample_locations = tf.linspace(tf.math.reduce_min(transformed_sampled_values), tf.math.reduce_max(transformed_sampled_values), self.pdf_resolution)

        # Sample_locations_fix offsets 0 valued sample location to prevent NaN values when used in truediv op
        sample_locations_fix = tf.where(tf.math.not_equal(sample_locations, [0.0]), sample_locations, [1.0e-7])
        transformed_pdf = transformed_dist.prob(sample_locations_fix)

        return transformed_pdf

    def get_config(self):
        config = super(KernelDensityEstimation, self).get_config()
        config.update({
            'trainable': self.trainable,
            'bank_capacity': self.bank_capacity,
            'bank_initializer': tf.keras.initializers.serialize(self.bank_initializer),
            'bandwidth': self.bandwidth,
            'pdf_resolution': self.pdf_resolution,
            'quantile_sample_resolution': self.quantile_sample_resolution,
            'random_state': self.random_state
        })
        return config


# Simple unit test in eager execution mode
# test_data_samples = tf.random.uniform((50, 142))
# custom_layer = KernelDensityEstimation(bank_capacity=20, bank_initializer='random_normal', bandwidth=0.335, dist_dim_nr=1, pdf_resolution=100, quantile_sample_resolution=1000, random_state=42)
# for _ in range(1):
#     output = custom_layer(test_data_samples)
#     print(output)


# Graph mode test
# input = tf.keras.Input(shape=(556, ), batch_size=12)
# output = KernelDensityEstimation(bank_capacity=40, bank_initializer='random_normal', bandwidth=1., pdf_resolution=50, quantile_sample_resolution=1000, dist_dim_nr=1, random_state=42)(input)
# model = tf.keras.models.Model(inputs=input, outputs=output)
# print(model.summary())


# Generalization to multivariate distributions
"""
# test_data_samples = tf.random.uniform((50, 1))
test_data_samples = tf.constant(0., shape=(50, 1))

initializer = tf.keras.initializers.he_uniform()
negative_example_bank = tf.Variable(initializer(shape=(40, 100 ** 1)), name="negative_example_bank")

kernel_iterator = lambda layer_input: tfp.distributions.Independent(tfp.distributions.MultivariateNormalDiag(loc=layer_input, scale_diag=[0.335] * layer_input.shape[-1]))
density_estimator = tfp.distributions.MixtureSameFamily(
    mixture_distribution=tfp.distributions.Categorical(probs=[1 / test_data_samples.shape[0]] * test_data_samples.shape[0]),
    components_distribution=kernel_iterator(test_data_samples)
)
quantiles = tfp.stats.quantiles(density_estimator.sample(sample_shape=1000), axis=0, num_quantiles=4, interpolation='nearest')

sample_locations = tf.transpose(tf.reshape(
    tf.meshgrid(*tf.unstack(tf.map_fn(fn=lambda x: tf.linspace(x[0], x[-1], 10), elems=tf.transpose(quantiles)), axis=0)),
    shape=(test_data_samples.shape[-1], -1)
))
current_pdf_flat = density_estimator.prob(sample_locations)
print(current_pdf_flat)
# print(negative_example_bank.sparse_read(0))
"""
