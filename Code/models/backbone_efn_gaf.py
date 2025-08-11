import tensorflow as tf


class EfficientNetGafBackbone:
    def __init__(self, hp, input_shape=(128, 2048, 6)):
        self.input_shape = input_shape
        self.hp = hp

        for shape_elem in input_shape[:-1]:
            if shape_elem % 2 != 0:
                raise ValueError("Input width or height should be divisible by 2.")

    def get_model(self):
        inp = tf.keras.layers.Input(shape=self.input_shape, name="channel_wise_dfs_input")

        # --------- Convolution layers --------- #
        inp_conv = tf.keras.layers.Conv2D(
            filters=20,
            kernel_size=(26, 14),
            strides=(1, 1),
            activation=None,
            padding="same",
            kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0, mode='fan_out', distribution='normal'),
            use_bias=False,
            name="channel_wise_dfs_inp_conv"
        )

        outp_conv = tf.keras.layers.Conv2D(
            filters=160,
            kernel_size=(6, 12),
            strides=(1, 1),
            activation=None,
            padding="same",
            kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0, mode='fan_out', distribution='normal'),
            use_bias=False,
            name="channel_wise_dfs_outp_conv"
        )

        d_inp_conv = tf.keras.layers.Conv2D(
            filters=20,
            kernel_size=(14, 30),
            activation=None,
            padding="same",
            kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0, mode='fan_out', distribution='normal'),
            use_bias=False,
            name="d_channel_wise_dfs_inp_conv"
        )

        d_inp_conv_2 = tf.keras.layers.Conv2D(
            filters=self.input_shape[-1],
            kernel_size=(42, 12),
            activation=None,
            padding="same",
            kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0, mode='fan_out', distribution='normal'),
            use_bias=False,
            name="d_channel_wise_dfs_outp_conv"
        )

        # --------- Encoder --------- #
        enc = inp_conv(inp)
        enc = tf.keras.layers.Activation(tf.keras.activations.swish)(enc)
        enc = tf.keras.layers.MaxPool2D((28, 38), strides=2, padding='same')(enc)

        # mbConv block 1
        enc = self.mbConv_block(
            kernel_size=6,
            inp_filters=20,
            outp_filters=40,
            exp_ratio=4,
            id_skip=False,
            strides=1,
            se_ratio=0.4,
            drop_rate=None,
            batch_norm=False
        )(enc)
        enc = tf.keras.layers.MaxPool2D((22, 22), strides=2, padding='same')(enc)

        # mbConv block 2
        enc = self.mbConv_block(
            kernel_size=3,
            inp_filters=40,
            outp_filters=80,
            exp_ratio=7,
            id_skip=False,
            strides=1,
            se_ratio=0.7,
            drop_rate=None,
            batch_norm=False
        )(enc)
        enc = tf.keras.layers.MaxPool2D((40, 14), strides=2, padding='same')(enc)

        enc = outp_conv(enc)
        e = tf.keras.layers.Activation(tf.keras.activations.swish)(enc)
        pre_pool_size = e._type_spec.shape[1:3]
        gmp_enc = tf.keras.layers.GlobalMaxPooling2D()(e)

        # --------- Decoder --------- #
        dec = tf.keras.layers.Reshape(target_shape=(1, 1, gmp_enc._type_spec.shape[-1]))(gmp_enc)
        dec = tf.keras.layers.UpSampling2D(size=pre_pool_size, interpolation='nearest')(dec)

        # mbDeconv block 1
        dec = self.mbDeconv_block(
            kernel_size=3,
            inp_filters=80,
            outp_filters=40,
            exp_ratio=7,
            id_skip=False,
            strides=1,
            se_ratio=0.7,
            drop_rate=None,
            batch_norm=False
        )(dec)
        dec = tf.keras.layers.UpSampling2D(size=2, interpolation='nearest')(dec)

        # mbDeconv block 2
        dec = self.mbDeconv_block(
            kernel_size=6,
            inp_filters=40,
            outp_filters=20,
            exp_ratio=4,
            id_skip=False,
            strides=1,
            se_ratio=0.4,
            drop_rate=None,
            batch_norm=False
        )(dec)
        dec = tf.keras.layers.UpSampling2D(size=2, interpolation='nearest')(dec)

        dec = d_inp_conv(dec)
        dec = tf.keras.layers.Activation(tf.keras.activations.swish)(dec)
        dec = tf.keras.layers.UpSampling2D(size=2, interpolation='nearest')(dec)

        dec = d_inp_conv_2(dec)
        dec = tf.keras.layers.Activation(tf.keras.activations.swish)(dec)

        return tf.keras.models.Model(inp, [gmp_enc, enc, dec], name='EffNet_GAF')

    # See: https://medium.com/analytics-vidhya/image-classification-with-
    # efficientnet-better-performance-with-computational-efficiency-f480fdb00ac6
    def mbConv_block(self, kernel_size, inp_filters, outp_filters, exp_ratio, id_skip, strides, se_ratio, drop_rate,
                     batch_norm):

        def block(inputs):
            # Expansion
            x = tf.keras.layers.Conv2D(
                filters=inp_filters * exp_ratio,
                kernel_size=1,
                strides=1,
                padding='same',
                use_bias=False,
                activation=None,
                kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0, mode='fan_out',
                                                                         distribution='normal'))(inputs)
            if batch_norm:
                x = tf.keras.layers.BatchNormalization(axis=3)(x)
            x = tf.keras.layers.Activation(tf.keras.activations.swish)(x)

            # Depthwise convolution
            x = tf.keras.layers.DepthwiseConv2D(
                kernel_size=kernel_size,
                strides=strides,
                padding='same',
                use_bias=False,
                activation=None,
                depthwise_initializer=tf.keras.initializers.VarianceScaling(scale=2.0, mode='fan_out',
                                                                            distribution='normal'))(x)
            if batch_norm:
                x = tf.keras.layers.BatchNormalization(axis=3)(x)
            x = tf.keras.layers.Activation(tf.keras.activations.swish)(x)

            # Squeeze and excitation
            se = tf.keras.layers.GlobalAveragePooling2D()(x)
            se = tf.keras.layers.Reshape((1, 1, inp_filters * exp_ratio))(se)
            se = tf.keras.layers.Conv2D(
                filters=max(1, int(inp_filters * se_ratio)),
                kernel_size=1,
                strides=1,
                padding='same',
                use_bias=True,
                activation=tf.keras.activations.swish,
                kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0, mode='fan_out',
                                                                         distribution='normal'))(se)
            se = tf.keras.layers.Conv2D(
                filters=inp_filters * exp_ratio,
                kernel_size=1,
                strides=1,
                padding='same',
                use_bias=True,
                activation='sigmoid',
                kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0, mode='fan_out',
                                                                         distribution='normal'))(se)
            x = tf.keras.layers.Multiply()([x, se])

            # Output
            x = tf.keras.layers.Conv2D(
                filters=outp_filters,
                kernel_size=1,
                strides=1,
                padding='same',
                use_bias=False,
                activation=None,
                kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0, mode='fan_out',
                                                                         distribution='normal'))(x)
            if batch_norm:
                x = tf.keras.layers.BatchNormalization(axis=3)(x)

            if id_skip:
                if all(s == 1 for s in strides) and inp_filters == outp_filters:
                    if drop_rate:
                        x = tf.keras.layers.Dropout(drop_rate, noise_shape=(None, 1, 1, 1))(x)

                    if x._type_spec.shape != inputs._type_spec.shape:
                        inputs = tf.keras.layers.Conv2D(x._type_spec.shape[-1], (1, 1), padding='same', use_bias=False,
                                                        activation=None,
                                                        kernel_initializer=tf.keras.initializers.VarianceScaling(
                                                            scale=2.0, mode='fan_out', distribution='normal'))(inputs)

                    x = tf.keras.layers.Add()([x, inputs])

            return x

        return block

    def mbDeconv_block(self, kernel_size, inp_filters, outp_filters, exp_ratio, id_skip, strides, se_ratio, drop_rate,
                       batch_norm):

        def block(inputs):
            # Expansion
            x = tf.keras.layers.Conv2D(
                filters=inp_filters * exp_ratio,
                kernel_size=1,
                strides=1,
                padding='same',
                use_bias=False,
                activation=None,
                kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0, mode='fan_out',
                                                                         distribution='normal'))(inputs)
            if batch_norm:
                x = tf.keras.layers.BatchNormalization(axis=3)(x)
            x = tf.keras.layers.Activation(tf.keras.activations.swish)(x)

            # Depthwise convolution
            x = tf.keras.layers.UpSampling2D(strides)(x)
            x = tf.keras.layers.DepthwiseConv2D(
                kernel_size=kernel_size,
                padding='same',
                use_bias=False,
                activation=None,
                depthwise_initializer=tf.keras.initializers.VarianceScaling(scale=2.0, mode='fan_out',
                                                                            distribution='normal'))(x)
            if batch_norm:
                x = tf.keras.layers.BatchNormalization(axis=3)(x)
            x = tf.keras.layers.Activation(tf.keras.activations.swish)(x)

            # Squeeze and excitation
            se = tf.keras.layers.GlobalAveragePooling2D()(x)
            se = tf.keras.layers.Reshape((1, 1, inp_filters * exp_ratio))(se)
            se = tf.keras.layers.Conv2D(
                filters=max(1, int(inp_filters * se_ratio)),
                kernel_size=1,
                strides=1,
                padding='same',
                use_bias=True,
                activation=tf.keras.activations.swish,
                kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0, mode='fan_out',
                                                                         distribution='normal'))(se)
            se = tf.keras.layers.Conv2D(
                filters=inp_filters * exp_ratio,
                kernel_size=1,
                strides=1,
                padding='same',
                use_bias=True,
                activation='sigmoid',
                kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0, mode='fan_out',
                                                                         distribution='normal'))(se)
            x = tf.keras.layers.Multiply()([x, se])

            # Output
            x = tf.keras.layers.Conv2D(
                filters=outp_filters,
                kernel_size=1,
                strides=1,
                padding='same',
                use_bias=False,
                activation=None,
                kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0, mode='fan_out',
                                                                         distribution='normal'))(x)
            if batch_norm:
                x = tf.keras.layers.BatchNormalization(axis=3)(x)

            if id_skip:
                if all(s == 1 for s in strides) and inp_filters == outp_filters:
                    if drop_rate:
                        x = tf.keras.layers.Dropout(drop_rate, noise_shape=(None, 1, 1, 1))(x)

                    if x._type_spec.shape != inputs._type_spec.shape:
                        inputs = tf.keras.layers.Conv2D(x._type_spec.shape[-1], (1, 1), padding='same', use_bias=False,
                                                        activation=None,
                                                        kernel_initializer=tf.keras.initializers.VarianceScaling(
                                                            scale=2.0, mode='fan_out', distribution='normal'))(inputs)

                    x = tf.keras.layers.Add()([x, inputs])

            return x

        return block


# yolo = EfficientNetGafBackbone(input_shape=(128, 2048, 6), hp=None)
# print(yolo.get_model().summary(line_length=150))
