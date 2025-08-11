from typing import Tuple

import tensorflow as tf
import tensorflow_addons as tfa

from custom_items.data_specs import input_specs_dataset
from custom_items.metrics import MultiClassPrecision, MultiClassRecall
from models.domain_techniques.classifier_head import make_classifier_head


def get_build_model(input_shape: Tuple[int, int, int], output_shape: int, domain_shape: int, backbone):
    def build_model(hp):
        dense_initializers = tf.keras.initializers.VarianceScaling(scale=1. / 3., mode='fan_out',
                                                                   distribution='uniform')

        std_extractor = backbone(hp, input_shape=input_shape).get_model()
        inp = tf.keras.layers.Input(shape=input_shape)
        enc, _, _ = std_extractor(inp)

        # standard neurons for resnet and efficientnet dfs
        layers = [400, 300, 200]
        if backbone == 'vgg':
            layers = [4096, 1024, 512]
        elif backbone == 'efficientnet':
            # efficient net and gaf is 160 for widar and 276/125 for signfi
            if input_shape == input_specs_dataset['widar']['gaf'] \
                    or input_shape == input_specs_dataset['signfi']['gaf']:
                layers = [160, 150, 150]

        for nr_count in layers:
            enc = tf.keras.layers.Dense(nr_count, activation='relu', kernel_initializer=dense_initializers)(enc)

        out_domain = tf.keras.layers.Dense(domain_shape, activation='softmax', kernel_initializer=dense_initializers,
                                           name='domain_output')(enc)
        # task output
        outp = make_classifier_head(enc, dense_initializers, backbone, input_shape, output_shape)

        complete_model = tf.keras.models.Model(inp, outputs=[outp, out_domain])
        # loss weights, 10^-6
        complete_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                               loss=['categorical_crossentropy', 'categorical_crossentropy'],
                               loss_weights=[1, 10 ** -6],
                               metrics=
                               {'task_output': [tf.keras.metrics.CategoricalAccuracy(),
                                                MultiClassPrecision(num_classes=output_shape, average='weighted'),
                                                MultiClassRecall(num_classes=output_shape, average='weighted'),
                                                tfa.metrics.F1Score(num_classes=output_shape, average='weighted'),
                                                tfa.metrics.CohenKappa(num_classes=output_shape, sparse_labels=False)]})

        return complete_model

    return build_model
