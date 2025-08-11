import keras.layers
import tensorflow as tf

from custom_items.data_specs import input_specs_dataset


def make_classifier_head(enc, dense_initializers, backbone, input_shape, output_shape) -> keras.layers.Layer:
    """
    Create the classifier head of the techniques, which depends on the backbone and input and output shape
    :param enc: the encoder layer
    :param dense_initializers: kernel intializers for dense layers
    :param backbone: the backbone type
    :param input_shape:
    :param output_shape:
    :return:
    """
    layers = [400]
    if backbone == 'vgg':
        layers = [4096, 4096]
    elif backbone == 'efficientnet':
        # efficient net and gaf goes from 160 to 150 for widar and 276/125 for signfi
        if input_shape == input_specs_dataset['widar']['gaf']:
            layers = [128]
        elif input_shape == input_specs_dataset['signfi']['gaf']:
            layers = [output_shape]

    for nr_count in layers:
        enc = tf.keras.layers.Dense(nr_count, activation='relu', kernel_initializer=dense_initializers)(enc)

    out_class = tf.keras.layers.Dense(output_shape, activation='softmax', kernel_initializer=dense_initializers,
                                      name='task_output')(enc)

    return out_class
