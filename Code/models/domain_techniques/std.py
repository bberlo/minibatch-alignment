from typing import Tuple

import tensorflow as tf
import tensorflow_addons as tfa

from custom_items.data_specs import input_specs_dataset
from custom_items.metrics import MultiClassPrecision, MultiClassRecall
from models.domain_techniques.classifier_head import make_classifier_head


def get_build_model(input_shape: Tuple[int, int, int], output_shape: int, backbone):
    def build_model(hp):
        dense_initializers = tf.keras.initializers.VarianceScaling(scale=1. / 3., mode='fan_out',
                                                                   distribution='uniform')

        std_extractor = backbone(hp, input_shape=input_shape).get_model()
        inp = tf.keras.layers.Input(shape=input_shape)
        enc, _, _ = std_extractor(inp)

        outp = make_classifier_head(enc, dense_initializers, backbone, input_shape, output_shape)

        complete_model = tf.keras.models.Model(inp, outp)
        complete_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                               loss='categorical_crossentropy',
                               metrics=[tf.keras.metrics.CategoricalAccuracy(),
                                        MultiClassPrecision(num_classes=output_shape, average='weighted'),
                                        MultiClassRecall(num_classes=output_shape, average='weighted'),
                                        tfa.metrics.F1Score(num_classes=output_shape, average='weighted'),
                                        tfa.metrics.CohenKappa(num_classes=output_shape, sparse_labels=False)])

        return complete_model

    return build_model
