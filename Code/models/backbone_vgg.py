import keras_tuner
import tensorflow as tf


class BackboneVGG:
    def __init__(self, hp, input_shape):
        self.input_shape = input_shape
        self.hp = hp

        for shape_elem in input_shape[:-1]:
            if shape_elem % 2 != 0:
                raise ValueError("Input width or height should be divisible by 2.")

    def get_model(self):
        inp = tf.keras.layers.Input(shape=self.input_shape, name="channel_wise_dfs_input")

        # ----Conv layers----

        # 1st conv block
        inp_conv = tf.keras.layers.Conv2D(filters=64,
                                          kernel_size=(
                                              self.hp.Int('kernel_size_conv_1_i', min_value=2, max_value=7, step=2,
                                                          default=3),
                                              self.hp.Int('kernel_size_conv_1_j', min_value=2, max_value=7, step=2,
                                                          default=3)),
                                          padding='same',
                                          kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0,
                                                                                                   mode='fan_out',
                                                                                                   distribution='normal'),
                                          activation='relu',
                                          name="channel_wise_dfs_inp_conv")(inp)

        inp_conv = tf.keras.layers.Conv2D(filters=64,
                                          kernel_size=(
                                              self.hp.Int('kernel_size_conv_1_i', min_value=2, max_value=7, step=2,
                                                          default=3),
                                              self.hp.Int('kernel_size_conv_1_j', min_value=2, max_value=7, step=2,
                                                          default=3)),
                                          padding='same',
                                          kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0,
                                                                                                   mode='fan_out',
                                                                                                   distribution='normal'),
                                          activation='relu',
                                          name="channel_wise_dfs_outp_conv_2")(inp_conv)

        inp_conv = tf.keras.layers.MaxPooling2D(pool_size=(
            self.hp.Int('pool_size_max_enc1_i', min_value=2, max_value=48, step=2, default=2),
            self.hp.Int('pool_size_max_enc1_j', min_value=2, max_value=48, step=2, default=2)),
            strides=2,
            padding='same')(inp_conv)

        # 2nd conv block
        inp_conv = tf.keras.layers.Conv2D(filters=128,
                                          kernel_size=(
                                              self.hp.Int('kernel_size_conv_2_i', min_value=2, max_value=7, step=2,
                                                          default=3),
                                              self.hp.Int('kernel_size_conv_2_j', min_value=2, max_value=7, step=2,
                                                          default=3)),
                                          padding='same',
                                          kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0,
                                                                                                   mode='fan_out',
                                                                                                   distribution='normal'),
                                          activation='relu',
                                          name="channel_wise_dfs_inp_conv_3")(inp_conv)

        inp_conv = tf.keras.layers.Conv2D(filters=128,
                                          kernel_size=(
                                              self.hp.Int('kernel_size_conv_2_i', min_value=2, max_value=7, step=2,
                                                          default=3),
                                              self.hp.Int('kernel_size_conv_2_j', min_value=2, max_value=7, step=2,
                                                          default=3)),
                                          padding='same',
                                          kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0,
                                                                                                   mode='fan_out',
                                                                                                   distribution='normal'),
                                          activation='relu',
                                          name="channel_wise_dfs_outp_conv_4")(inp_conv)

        inp_conv = tf.keras.layers.MaxPooling2D(pool_size=(
            self.hp.Int('pool_size_max_enc2_i', min_value=2, max_value=48, step=2, default=2),
            self.hp.Int('pool_size_max_enc2_j', min_value=2, max_value=48, step=2, default=2)),
            strides=2,
            padding='same')(inp_conv)

        # 3rd conv block
        inp_conv = tf.keras.layers.Conv2D(filters=256,
                                          kernel_size=(
                                              self.hp.Int('kernel_size_conv_3_i', min_value=2, max_value=7, step=2,
                                                          default=3),
                                              self.hp.Int('kernel_size_conv_3_j', min_value=2, max_value=7, step=2,
                                                          default=3)),
                                          padding='same',
                                          kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0,
                                                                                                   mode='fan_out',
                                                                                                   distribution='normal'),
                                          activation='relu',
                                          name="channel_wise_dfs_inp_conv_5")(inp_conv)
        inp_conv = tf.keras.layers.Conv2D(filters=256,
                                          kernel_size=(
                                              self.hp.Int('kernel_size_conv_3_i', min_value=2, max_value=7, step=2,
                                                          default=3),
                                              self.hp.Int('kernel_size_conv_3_j', min_value=2, max_value=7, step=2,
                                                          default=3)),
                                          padding='same',
                                          kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0,
                                                                                                   mode='fan_out',
                                                                                                   distribution='normal'),
                                          activation='relu',
                                          name="channel_wise_dfs_outp_conv_6")(inp_conv)
        inp_conv = tf.keras.layers.Conv2D(filters=256,
                                          kernel_size=(
                                              self.hp.Int('kernel_size_conv_4_i', min_value=2, max_value=7, step=2,
                                                          default=3),
                                              self.hp.Int('kernel_size_conv_4_j', min_value=2, max_value=7, step=2,
                                                          default=3)),
                                          padding='same',
                                          kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0,
                                                                                                   mode='fan_out',
                                                                                                   distribution='normal'),
                                          activation='relu',
                                          name="channel_wise_dfs_outp_conv_7")(inp_conv)
        inp_conv = tf.keras.layers.MaxPool2D(pool_size=(
            self.hp.Int('pool_size_max_enc3_i', min_value=2, max_value=48, step=2, default=2),
            self.hp.Int('pool_size_max_enc3_j', min_value=2, max_value=48, step=2, default=2)),
            strides=2,
            padding='same')(inp_conv)

        # 4th conv block
        inp_conv = tf.keras.layers.Conv2D(filters=512,
                                          kernel_size=(
                                              self.hp.Int('kernel_size_conv_4_i', min_value=2, max_value=7, step=2,
                                                          default=3),
                                              self.hp.Int('kernel_size_conv_4_j', min_value=2, max_value=7, step=2,
                                                          default=3)),
                                          padding='same',
                                          kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0,
                                                                                                   mode='fan_out',
                                                                                                   distribution='normal'),
                                          activation='relu',
                                          name="channel_wise_dfs_inp_conv_8")(inp_conv)

        inp_conv = tf.keras.layers.Conv2D(filters=512,
                                          kernel_size=(
                                              self.hp.Int('kernel_size_conv_4_i', min_value=2, max_value=7, step=2,
                                                          default=3),
                                              self.hp.Int('kernel_size_conv_4_j', min_value=2, max_value=7, step=2,
                                                          default=3)),
                                          padding='same',
                                          kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0,
                                                                                                   mode='fan_out',
                                                                                                   distribution='normal'),
                                          activation='relu',
                                          name="channel_wise_dfs_outp_conv_9")(inp_conv)
        inp_conv = tf.keras.layers.Conv2D(filters=512,
                                          kernel_size=(
                                              self.hp.Int('kernel_size_conv_4_i', min_value=2, max_value=7, step=2,
                                                          default=3),
                                              self.hp.Int('kernel_size_conv_4_j', min_value=2, max_value=7, step=2,
                                                          default=3)),
                                          padding='same',
                                          kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0,
                                                                                                   mode='fan_out',
                                                                                                   distribution='normal'),
                                          activation='relu',
                                          name="channel_wise_dfs_outp_conv_10")(inp_conv)
        inp_conv = tf.keras.layers.MaxPool2D(pool_size=(
            self.hp.Int('pool_size_max_enc4_i', min_value=2, max_value=48, step=2, default=2),
            self.hp.Int('pool_size_max_enc4_j', min_value=2, max_value=48, step=2, default=2)),
            strides=2,
            padding='same')(inp_conv)

        # 5th conv block
        inp_conv = tf.keras.layers.Conv2D(filters=512,
                                          kernel_size=(
                                              self.hp.Int('kernel_size_conv_5_i', min_value=2, max_value=7, step=2,
                                                          default=3),
                                              self.hp.Int('kernel_size_conv_5_j', min_value=2, max_value=7, step=2,
                                                          default=3)),
                                          padding='same',
                                          kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0,
                                                                                                   mode='fan_out',
                                                                                                   distribution='normal'),
                                          activation='relu',
                                          name="channel_wise_dfs_inp_conv_11")(inp_conv)
        inp_conv = tf.keras.layers.Conv2D(filters=512,
                                          kernel_size=(
                                              self.hp.Int('kernel_size_conv_5_i', min_value=2, max_value=7, step=2,
                                                          default=3),
                                              self.hp.Int('kernel_size_conv_5_j', min_value=2, max_value=7, step=2,
                                                          default=3)),
                                          padding='same',
                                          kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0,
                                                                                                   mode='fan_out',
                                                                                                   distribution='normal'),
                                          activation='relu',
                                          name="channel_wise_dfs_outp_conv_12")(inp_conv)
        inp_conv = tf.keras.layers.Conv2D(filters=512,
                                          kernel_size=(
                                              self.hp.Int('kernel_size_conv_5_i', min_value=2, max_value=7, step=2,
                                                          default=3),
                                              self.hp.Int('kernel_size_conv_5_j', min_value=2, max_value=7, step=2,
                                                          default=3)),
                                          padding='same',
                                          kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0,
                                                                                                   mode='fan_out',
                                                                                                   distribution='normal'),
                                          activation='relu',
                                          name="channel_wise_dfs_outp_conv_13")(inp_conv)
        inp_conv = tf.keras.layers.MaxPool2D(pool_size=(
            self.hp.Int('pool_size_max_enc5_i', min_value=2, max_value=48, step=2, default=2),
            self.hp.Int('pool_size_max_enc5_j', min_value=2, max_value=48, step=2, default=2)),
            strides=2,
            padding='same')(inp_conv)

        pre_flatten_shape = inp_conv._type_spec.shape[1:]
        inp_conv_o = tf.keras.layers.Flatten()(inp_conv)

        # Decoder part
        d_inp_conv = tf.keras.layers.Reshape(target_shape=pre_flatten_shape)(inp_conv_o)

        # First decode block
        d_inp_conv = tf.keras.layers.UpSampling2D()(d_inp_conv)
        d_inp_conv = tf.keras.layers.Conv2D(filters=512, kernel_size=(
            self.hp.Int('kernel_size_conv_5_i', min_value=2, max_value=7, step=2, default=3),
            self.hp.Int('kernel_size_conv_5_j', min_value=2, max_value=7, step=2, default=3)), padding='same',
                                          kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0,
                                                                                                   mode='fan_out',
                                                                                                   distribution='normal'),
                                          activation='relu', name="d_channel_wise_dfs_inp_conv_13")(d_inp_conv)
        d_inp_conv = tf.keras.layers.Conv2D(filters=512, kernel_size=(
            self.hp.Int('kernel_size_conv_5_i', min_value=2, max_value=7, step=2, default=3),
            self.hp.Int('kernel_size_conv_5_j', min_value=2, max_value=7, step=2, default=3)), padding='same',
                                          kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0,
                                                                                                   mode='fan_out',
                                                                                                   distribution='normal'),
                                          activation='relu', name="d_channel_wise_dfs_outp_conv_12")(d_inp_conv)
        d_inp_conv = tf.keras.layers.Conv2D(filters=512, kernel_size=(
            self.hp.Int('kernel_size_conv_5_i', min_value=2, max_value=7, step=2, default=3),
            self.hp.Int('kernel_size_conv_5_j', min_value=2, max_value=7, step=2, default=3)), padding='same',
                                          kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0,
                                                                                                   mode='fan_out',
                                                                                                   distribution='normal'),
                                          activation='relu', name="d_channel_wise_dfs_outp_conv_11")(d_inp_conv)

        # Second decode block
        d_inp_conv = tf.keras.layers.UpSampling2D()(d_inp_conv)
        d_inp_conv = tf.keras.layers.Conv2D(filters=512, kernel_size=(
            self.hp.Int('kernel_size_conv_5_i', min_value=2, max_value=7, step=2, default=3),
            self.hp.Int('kernel_size_conv_5_j', min_value=2, max_value=7, step=2, default=3)), padding='same',
                                            kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0,
                                                                                                     mode='fan_out',
                                                                                                     distribution='normal'),
                                            activation='relu', name="d_channel_wise_dfs_inp_conv_10")(d_inp_conv)
        d_inp_conv = tf.keras.layers.Conv2D(filters=512, kernel_size=(
            self.hp.Int('kernel_size_conv_5_i', min_value=2, max_value=7, step=2, default=3),
            self.hp.Int('kernel_size_conv_5_j', min_value=2, max_value=7, step=2, default=3)), padding='same',
                                            kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0,
                                                                                                     mode='fan_out',
                                                                                                     distribution='normal'),
                                            activation='relu', name="d_channel_wise_dfs_outp_conv_9")(d_inp_conv)
        d_inp_conv = tf.keras.layers.Conv2D(filters=512, kernel_size=(
            self.hp.Int('kernel_size_conv_5_i', min_value=2, max_value=7, step=2, default=3),
            self.hp.Int('kernel_size_conv_5_j', min_value=2, max_value=7, step=2, default=3)), padding='same',
                                            kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0,
                                                                                                     mode='fan_out',
                                                                                                     distribution='normal'),
                                            activation='relu', name="d_channel_wise_dfs_outp_conv_8")(d_inp_conv)

        # Third decode block
        d_inp_conv = tf.keras.layers.UpSampling2D()(d_inp_conv)
        d_inp_conv = tf.keras.layers.Conv2D(filters=256, kernel_size=(
            self.hp.Int('kernel_size_conv_3_i', min_value=2, max_value=7, step=2, default=3),
            self.hp.Int('kernel_size_conv_3_j', min_value=2, max_value=7, step=2, default=3)), padding='same',
                                          kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0,
                                                                                                   mode='fan_out',
                                                                                                   distribution='normal'),
                                          activation='relu', name="d_channel_wise_dfs_inp_conv_7")(d_inp_conv)
        d_inp_conv = tf.keras.layers.Conv2D(filters=256, kernel_size=(
            self.hp.Int('kernel_size_conv_3_i', min_value=2, max_value=7, step=2, default=3),
            self.hp.Int('kernel_size_conv_3_j', min_value=2, max_value=7, step=2, default=3)), padding='same',
                                          kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0,
                                                                                                   mode='fan_out',
                                                                                                   distribution='normal'),
                                          activation='relu', name="d_channel_wise_dfs_outp_conv_6")(d_inp_conv)
        d_inp_conv = tf.keras.layers.Conv2D(filters=256, kernel_size=(
            self.hp.Int('kernel_size_conv_4_i', min_value=2, max_value=7, step=2, default=3),
            self.hp.Int('kernel_size_conv_4_j', min_value=2, max_value=7, step=2, default=3)), padding='same',
                                          kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0,
                                                                                                   mode='fan_out',
                                                                                                   distribution='normal'),
                                          activation='relu', name="d_channel_wise_dfs_outp_conv_5")(d_inp_conv)

        # Fourth decode block
        d_inp_conv = tf.keras.layers.UpSampling2D()(d_inp_conv)
        d_inp_conv = tf.keras.layers.Conv2D(filters=128, kernel_size=(
            self.hp.Int('kernel_size_conv_2_i', min_value=2, max_value=7, step=2, default=3),
            self.hp.Int('kernel_size_conv_2_j', min_value=2, max_value=7, step=2, default=3)), padding='same',
                                          kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0,
                                                                                                   mode='fan_out',
                                                                                                   distribution='normal'),
                                          activation='relu', name="d_channel_wise_dfs_inp_conv_4")(d_inp_conv)

        d_inp_conv = tf.keras.layers.Conv2D(filters=128, kernel_size=(
            self.hp.Int('kernel_size_conv_2_i', min_value=2, max_value=7, step=2, default=3),
            self.hp.Int('kernel_size_conv_2_j', min_value=2, max_value=7, step=2, default=3)), padding='same',
                                          kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0,
                                                                                                   mode='fan_out',
                                                                                                   distribution='normal'),
                                          activation='relu', name="d_channel_wise_dfs_outp_conv_3")(d_inp_conv)

        # Fifth decode block
        d_inp_conv = tf.keras.layers.UpSampling2D()(d_inp_conv)
        d_inp_conv = tf.keras.layers.Conv2D(filters=64, kernel_size=(
            self.hp.Int('kernel_size_conv_1_i', min_value=2, max_value=7, step=2, default=3),
            self.hp.Int('kernel_size_conv_1_j', min_value=2, max_value=7, step=2, default=3)), padding='same',
                                          kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0,
                                                                                                   mode='fan_out',
                                                                                                   distribution='normal'),
                                          activation='relu', name="d_channel_wise_dfs_inp_conv_2")(d_inp_conv)

        d_inp_conv = tf.keras.layers.Conv2D(filters=64, kernel_size=(
            self.hp.Int('kernel_size_conv_1_i', min_value=2, max_value=7, step=2, default=3),
            self.hp.Int('kernel_size_conv_1_j', min_value=2, max_value=7, step=2, default=3)), padding='same',
                                          kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0,
                                                                                                   mode='fan_out',
                                                                                                   distribution='normal'),
                                          activation='relu', name="d_channel_wise_dfs_inp_conv")(d_inp_conv)

        # Output decode block
        d_inp_conv = tf.keras.layers.Conv2D(filters=self.input_shape[-1], kernel_size=(
            self.hp.Int('kernel_size_conv_1_i', min_value=2, max_value=7, step=2, default=3),
            self.hp.Int('kernel_size_conv_1_j', min_value=2, max_value=7, step=2, default=3)), padding='same',
                                          kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0,
                                                                                                   mode='fan_out',
                                                                                                   distribution='normal'),
                                          activation='relu', name="d_channel_wise_dfs_outp_conv")(d_inp_conv)

        return tf.keras.models.Model(inp, [inp_conv_o, inp_conv, d_inp_conv], name='vgg')


# default_vgg_hps = keras_tuner.HyperParameters()
# yolo = BackboneVGG(default_vgg_hps, input_shape=(128, 2048, 6))
# print(yolo.get_model().summary(line_length=150))
