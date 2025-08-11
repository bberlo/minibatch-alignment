import keras_tuner
import tensorflow as tf


class BackboneResNet:
    def __init__(self, hp, input_shape):
        self.input_shape = input_shape
        self.hp = hp

        for shape_elem in input_shape[:-1]:
            if shape_elem % 2 != 0:
                raise ValueError("Input width or height should be divisible by 2.")

    def get_model(self):
        # step 1 (setup input layer)
        inp = tf.keras.layers.Input(shape=self.input_shape, name="channel_wise_dfs_input")

        # Step 2 (initial conv layer along with maxpool)
        inp_conv = tf.keras.layers.Conv2D(filters=64,
                                          kernel_size=(
                                              self.hp.Int('kernel_size_conv_conv_i', min_value=2, max_value=7, step=2,
                                                          default=7),
                                              self.hp.Int('kernel_size_conv_conv_j', min_value=2, max_value=7, step=2,
                                                          default=7)),
                                          strides=2,
                                          padding='same',
                                          kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0,
                                                                                                   mode='fan_out',
                                                                                                   distribution='normal'),
                                          name="channel_wise_dfs_inp_conv")(inp)

        inp_conv = tf.keras.layers.BatchNormalization()(inp_conv)
        inp_conv = tf.keras.layers.Activation('relu')(inp_conv)
        inp_conv = tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding="same")(inp_conv)

        # Define size of sub_block and initial filter size
        block_layers = [2, 2, 2, 2]
        filter_size = 64

        # Step 3 Add resnet blocks
        def identity_block(x, filter):
            # copy tensor to variable called x_skip
            x_skip = x
            # Layer 1
            x = tf.keras.layers.Conv2D(filter, (3, 3), padding='same')(x)
            x = tf.keras.layers.BatchNormalization(axis=3)(x)
            x = tf.keras.layers.Activation('relu')(x)
            # Layer 2
            x = tf.keras.layers.Conv2D(filter, (3, 3), padding='same')(x)
            x = tf.keras.layers.BatchNormalization(axis=3)(x)
            # Add Residue
            x = tf.keras.layers.Add()([x, x_skip])
            x = tf.keras.layers.Activation('relu')(x)
            return x

        def convolutional_block(x, filter):
            # copy tensor to variable called x_skip
            x_skip = x
            # Layer 1
            x = tf.keras.layers.Conv2D(filter, (3, 3), padding='same', strides=(2, 2))(x)
            x = tf.keras.layers.BatchNormalization(axis=3)(x)
            x = tf.keras.layers.Activation('relu')(x)
            # Layer 2
            x = tf.keras.layers.Conv2D(filter, (3, 3), padding='same')(x)
            x = tf.keras.layers.BatchNormalization(axis=3)(x)
            # Processing Residue with conv(1,1)
            x_skip = tf.keras.layers.Conv2D(filter, (1, 1), padding='same', strides=(2, 2))(x_skip)
            # Add Residue
            x = tf.keras.layers.Add()([x, x_skip])
            x = tf.keras.layers.Activation('relu')(x)
            return x

        for i in range(4):
            if i == 0:
                # For sub-block 1 Residual/Convolutional block not needed
                for j in range(block_layers[i]):
                    inp_conv = identity_block(inp_conv, filter_size)
            else:
                # One Residual/Convolutional Block followed by Identity blocks
                # The filter size will go on increasing by a factor of 2
                filter_size = filter_size * 2
                inp_conv = convolutional_block(inp_conv, filter_size)
                for j in range(block_layers[i] - 1):
                    inp_conv = identity_block(inp_conv, filter_size)

        pre_pool_size = inp_conv._type_spec.shape[1:3]
        inp_conv_o = tf.keras.layers.GlobalAveragePooling2D()(inp_conv)

        # ResNet decoder part
        # Upsample-Conv used to prevent checkerboard artifacts: https://distill.pub/2016/deconv-checkerboard/
        def deconvolutional_block(x, filter):
            # copy tensor to variable called x_skip
            x_skip = x
            # Layer 1
            x = tf.keras.layers.UpSampling2D()(x)
            x = tf.keras.layers.Conv2D(filter, (3, 3), padding='same')(x)
            x = tf.keras.layers.BatchNormalization(axis=3)(x)
            x = tf.keras.layers.Activation('relu')(x)
            # Layer 2
            x = tf.keras.layers.Conv2D(filter, (3, 3), padding='same')(x)
            x = tf.keras.layers.BatchNormalization(axis=3)(x)
            # Processing Residue with conv(1,1)
            x_skip = tf.keras.layers.UpSampling2D()(x_skip)
            x_skip = tf.keras.layers.Conv2D(filter, (1, 1), padding='same')(x_skip)
            # Add Residue
            x = tf.keras.layers.Add()([x, x_skip])
            x = tf.keras.layers.Activation('relu')(x)
            return x

        # Inspiration: https://stats.stackexchange.com/questions/471345/inverse-of-global-average-pooling
        d_inp_conv = tf.keras.layers.Reshape(target_shape=(1, 1, inp_conv_o._type_spec.shape[-1]))(inp_conv_o)
        d_inp_conv = tf.keras.layers.UpSampling2D(size=pre_pool_size, interpolation='nearest')(d_inp_conv)

        for i in range(4):
            if i == list(range(4))[-1]:
                for j in range(block_layers[i]):
                    d_inp_conv = identity_block(d_inp_conv, filter_size)
            else:
                filter_size = filter_size // 2
                d_inp_conv = deconvolutional_block(d_inp_conv, filter_size)
                for j in range(block_layers[i] - 1):
                    d_inp_conv = identity_block(d_inp_conv, filter_size)

        # MaxPool2D inverse approximation
        d_inp_conv = tf.keras.layers.UpSampling2D()(d_inp_conv)
        d_inp_conv = tf.keras.layers.Conv2D(filters=64,
                                            kernel_size=(
            self.hp.Int('kernel_size_conv_conv_i', min_value=2, max_value=7, step=2, default=7),
            self.hp.Int('kernel_size_conv_conv_j', min_value=2, max_value=7, step=2, default=7)),
                                            padding='same',
                                            kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0,
                                                                                                     mode='fan_out',
                                                                                                     distribution='normal'))(d_inp_conv)
        d_inp_conv = tf.keras.layers.BatchNormalization()(d_inp_conv)
        d_inp_conv = tf.keras.layers.Activation('relu')(d_inp_conv)

        d_inp_conv = tf.keras.layers.UpSampling2D()(d_inp_conv)
        d_inp_conv = tf.keras.layers.Conv2D(filters=self.input_shape[-1],
                                            kernel_size=(
            self.hp.Int('kernel_size_conv_conv_i', min_value=2, max_value=7, step=2, default=7),
            self.hp.Int('kernel_size_conv_conv_j', min_value=2, max_value=7, step=2, default=7)),
                                            padding='same',
                                            kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0,
                                                                                                     mode='fan_out',
                                                                                                     distribution='normal'))(d_inp_conv)
        d_inp_conv = tf.keras.layers.BatchNormalization()(d_inp_conv)
        d_inp_conv = tf.keras.layers.Activation('relu')(d_inp_conv)

        return tf.keras.models.Model(inputs=inp, outputs=[inp_conv_o, inp_conv, d_inp_conv], name="resnet18")


# default_resnet_hps = keras_tuner.HyperParameters()
# yolo = BackboneResNet(default_resnet_hps, input_shape=(128, 2048, 6))
# print(yolo.get_model().summary(line_length=150))

# default_resnet_hps.Fixed('kernel_size_conv_conv_i', 7)
# default_resnet_hps.Fixed('kernel_size_conv_conv_j', 7)
