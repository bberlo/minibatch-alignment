from typing import Tuple

import tensorflow as tf
import tensorflow_addons as tfa

from custom_items.metrics import MultiClassPrecision, MultiClassRecall
from models.domain_techniques.classifier_head import make_classifier_head


def get_build_model(input_shape: Tuple[int, int, int], output_shape: int, backbone):
    def build_model(hp):
        kernel_initializers = tf.keras.initializers.VarianceScaling(scale=2.0, mode='fan_out', distribution='normal')
        dense_initializers = tf.keras.initializers.VarianceScaling(scale=1. / 3., mode='fan_out',
                                                                   distribution='uniform')

        backbone_block = backbone(hp, input_shape=input_shape).get_model()

        inp = tf.keras.layers.Input(shape=input_shape)
        enc = tf.keras.layers.Layer()(inp)
        # attention block a
        attention_a = attention_module_a(enc, hp, kernel_initializers)
        # backbone part
        _, enc, _ = backbone_block(attention_a)
        # attention block b
        attention_b = attention_module_b(enc, hp, kernel_initializers)
        # global pooling
        enc_o = tf.keras.layers.GlobalMaxPooling2D()(attention_b)

        # dense classifier part
        outp = make_classifier_head(enc_o, dense_initializers, backbone, input_shape, output_shape)

        complete_model = tf.keras.models.Model(inp, outp)
        complete_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                               loss='categorical_crossentropy',
                               metrics=[tf.keras.metrics.CategoricalAccuracy(),
                                        MultiClassPrecision(num_classes=output_shape, average='weighted'),
                                        MultiClassRecall(num_classes=output_shape, average='weighted'),
                                        tfa.metrics.F1Score(num_classes=output_shape, average='weighted'),
                                        tfa.metrics.CohenKappa(num_classes=output_shape, sparse_labels=False)])

        return complete_model

    def attention_module_a(enc, hp, kernel_initializers):
        # --------- Attention module A --------- #
        att_conv_a = tf.keras.layers.Conv2D(
            filters=1,
            kernel_size=(hp.Int('kernel_size_conv_att_a_i', min_value=16, max_value=88, step=8, default=48),
                         hp.Int('kernel_size_conv_att_a_j', min_value=16, max_value=88, step=8, default=32)),
            strides=(1, 1),
            activation=None,
            padding="same",
            kernel_initializer=kernel_initializers,
            use_bias=False,
            name="att_conv_a"
        )

        att_inp = tf.keras.layers.BatchNormalization()(enc)
        att_inp = att_conv_a(att_inp)
        att_inp = tf.keras.layers.Activation(tf.keras.activations.sigmoid)(att_inp)

        inp_mult = tf.keras.layers.Multiply()([att_inp, enc])
        enc = tf.keras.layers.Add()([inp_mult, enc])

        return enc

    def shared_mlp(hp, backbone_outp_shape, kernel_initializers):
        def block(inputs):
            x = tf.keras.layers.Flatten()(inputs)
            neurons_layers = [1184, 624]

            for neurons in neurons_layers:
                x = tf.keras.layers.Dense(neurons, activation='relu', kernel_initializer=kernel_initializers)(x)

            x = tf.keras.layers.Dense(backbone_outp_shape[-1], activation='relu',
                                      kernel_initializer=kernel_initializers)(x)
            x = tf.keras.layers.Reshape(tuple([*(1,) * len(backbone_outp_shape[:-1]), backbone_outp_shape[-1]]))(x)

            return x

        return block

    def attention_module_b(enc, hp, kernel_initializers):
        # --------- Attention module B --------- #
        att_mlp_b = shared_mlp(hp, enc.shape[1:], kernel_initializers)
        stride_i_mlp_b = hp.Int('strides_mlp_b_enc_i', min_value=16, max_value=32, step=2, default=22)
        stride_j_mlp_b = hp.Int('strides_mlp_b_enc_j', min_value=16, max_value=32, step=2, default=26)

        avg_enc = tf.keras.layers.AveragePooling2D(pool_size=(
            hp.Int('pool_size_avg_enc_i', min_value=8, max_value=48, step=2, default=22),
            hp.Int('pool_size_avg_enc_j', min_value=8, max_value=48, step=2, default=20)),
            strides=(stride_i_mlp_b, stride_j_mlp_b), padding='same')(enc)
        avg_enc = att_mlp_b(avg_enc)

        max_enc = tf.keras.layers.MaxPooling2D(pool_size=(
            hp.Int('pool_size_max_enc_i', min_value=8, max_value=48, step=2, default=28),
            hp.Int('pool_size_max_enc_j', min_value=8, max_value=48, step=2, default=14)),
            strides=(stride_i_mlp_b, stride_j_mlp_b), padding='same')(enc)
        max_enc = att_mlp_b(max_enc)

        comb_enc = tf.keras.layers.Add()([avg_enc, max_enc])
        comb_enc = tf.keras.layers.Activation(tf.keras.activations.sigmoid)(comb_enc)

        enc_interm = tf.keras.layers.Multiply()([comb_enc, enc])

        return enc_interm

    return build_model
