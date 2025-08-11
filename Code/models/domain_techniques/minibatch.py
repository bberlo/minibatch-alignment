from typing import Tuple

import tensorflow as tf
import tensorflow_addons as tfa

from custom_items.Kernel_density_estimation_layer import KernelDensityEstimation
from custom_items.Kernel_density_estimation_loss import SimilarityCrossentropy
from custom_items.Kernel_density_estimation_similarities import BilinearSimilarity
from custom_items.metrics import MultiClassPrecision, MultiClassRecall
from models.domain_techniques.classifier_head import make_classifier_head

settings_minibatch = {
    'bank_capacity': 40,
    'kernel_bandwidth': 0.335,
    'pdf_resolution': 100,
    'quantile_resolution': 1000,
    'domain_loss_weight': 10 ** -8,
}


def get_build_model(input_shape: Tuple[int, int, int], output_shape: int, backbone, batch_size: int = 12):
    """
    Settings to build model
    :param input_shape:
    :param output_shape:
    :param backbone:
    :param batch_size: the batch size must be fixed for minibatch
    :return:
    """

    def build_model(hp):
        dense_initializers = tf.keras.initializers.VarianceScaling(scale=1. / 3., mode='fan_out',
                                                                   distribution='uniform')

        std_extractor = backbone(hp, input_shape=input_shape).get_model()
        inp = tf.keras.layers.Input(shape=input_shape, batch_size=batch_size)

        if backbone == 'vgg':
            _, enc_not_flattend, _ = std_extractor(inp)

            # global pooling because otherwise input is to large
            enc = tf.keras.layers.GlobalMaxPooling2D()(enc_not_flattend)
        else:
            enc, _, _ = std_extractor(inp)

        # task output
        outp = make_classifier_head(enc, dense_initializers, backbone, input_shape, output_shape)

        x_2 = tf.keras.layers.Concatenate()([enc, outp])
        x_2 = KernelDensityEstimation(bank_capacity=settings_minibatch['bank_capacity'],
                                      bank_initializer=dense_initializers,
                                      bandwidth=settings_minibatch['kernel_bandwidth'],
                                      pdf_resolution=settings_minibatch['pdf_resolution'],
                                      quantile_sample_resolution=settings_minibatch['quantile_resolution'],
                                      random_state=42)(x_2)
        o_2 = BilinearSimilarity(
            initializer=dense_initializers,
            kernel_regularizer=None)(x_2)

        complete_model = tf.keras.models.Model(inp, [outp, o_2], name='minibatch')
        complete_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                               loss=[tf.keras.losses.CategoricalCrossentropy(),
                                     SimilarityCrossentropy(batch_size=batch_size)],
                               loss_weights=[1.0, -1.0 * settings_minibatch['domain_loss_weight']],
                               metrics={'task_output': [tf.keras.metrics.CategoricalAccuracy(),
                                                        MultiClassPrecision(num_classes=output_shape,
                                                                            average='weighted'),
                                                        MultiClassRecall(num_classes=output_shape, average='weighted'),
                                                        tfa.metrics.F1Score(num_classes=output_shape,
                                                                            average='weighted'),
                                                        tfa.metrics.CohenKappa(num_classes=output_shape,
                                                                               sparse_labels=False)]})
        return complete_model

    return build_model
