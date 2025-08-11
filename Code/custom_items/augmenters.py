import tensorflow as tf

from custom_items.data_specs import input_specs_dataset


# Augmenter methods are taken from 'Data Augmentation and Dense-LSTM for Human Activity Recognition Using WiFi Signal':
# https://doi.org/10.1109/JIOT.2020.3026732
# Note: augmenter methods have been altered to support two direction dfs in which 0 Hz is centered in the middle


# Custom dropout implementation to avoid non-zero scaling
def dropout(x, rate, seed):
    def func_zero():
        return x

    def func_not_zero():
        random_tensor = tf.random.uniform(shape=x.shape, minval=0, maxval=1, dtype=tf.float32, seed=seed)
        keep_mask = random_tensor >= tf.convert_to_tensor(rate, dtype=tf.float32)
        return tf.multiply(x, tf.cast(keep_mask, dtype=x.dtype))

    ret = tf.cond(tf.equal(tf.convert_to_tensor(rate, dtype=tf.float32), tf.constant(0.)), true_fn=func_zero,
                  false_fn=func_not_zero)

    return ret


def time_stretch(x, rate):
    if len(x.shape) == 4:
        shape = tf.slice(x.shape, begin=[1], size=[2])
    else:
        shape = tf.slice(x.shape, begin=[0], size=[2])

    scale_factors = tf.tensor_scatter_nd_update(
        tf.ones_like(shape, dtype=tf.float32),
        tf.constant([[1]]),
        tf.add(tf.constant([1.]), tf.expand_dims(tf.convert_to_tensor(rate), axis=-1)))
    new_shape = tf.multiply(tf.cast(shape, dtype=tf.float32), scale_factors)

    ret = tf.image.resize(x, tf.cast(new_shape, dtype=tf.int32), method='bilinear')
    ret = tf.image.resize_with_crop_or_pad(ret, target_height=shape[0], target_width=shape[1])

    return ret


def spec_shift(x, shift_amount):
    if len(x.shape) == 4:
        x = tf.concat([x[:, :x.shape[1] // 2, :, :], tf.reverse(x[:, x.shape[1] // 2:, :, :], axis=[1])], axis=2)
        to_begin = tf.scatter_nd(tf.constant([[1]]), tf.constant([5]), tf.constant([4]))
        the_size = tf.tensor_scatter_nd_update(x.shape, tf.constant([[1]]), tf.constant([40]))
    else:
        x = tf.concat([x[:x.shape[0] // 2, :, :], tf.reverse(x[x.shape[0] // 2:, :, :], axis=[0])], axis=1)
        to_begin = tf.scatter_nd(tf.constant([[0]]), tf.constant([5]), tf.constant([3]))
        the_size = tf.tensor_scatter_nd_update(x.shape, tf.constant([[0]]), tf.constant([40]))

    to_shift = tf.slice(x, to_begin, the_size)

    if len(x.shape) == 4:
        ret = tf.transpose(tf.tensor_scatter_nd_update(
            tf.transpose(x, perm=[1, 0, 2, 3]),
            tf.expand_dims(tf.range(5 + shift_amount, 40 + 5 + shift_amount, delta=1), axis=-1),
            tf.transpose(to_shift, perm=[1, 0, 2, 3])), perm=[1, 0, 2, 3])
        ret = tf.concat([ret[:, :, :ret.shape[2] // 2, :], tf.reverse(ret[:, :, ret.shape[2] // 2:, :], axis=[1])],
                        axis=1)
    else:
        ret = tf.tensor_scatter_nd_update(
            x, tf.expand_dims(tf.range(5 + shift_amount, 40 + 5 + shift_amount, delta=1), axis=-1), to_shift)
        ret = tf.concat([ret[:, :ret.shape[1] // 2, :], tf.reverse(ret[:, ret.shape[1] // 2:, :], axis=[0])], axis=0)

    return ret


def scale(x, rate):
    return tf.math.scalar_mul(1. + rate, x)


def freq_filters(x, delta, seed, min_freq, max_freq):
    mu_neg = tf.cast(tf.random.uniform(shape=(), minval=min_freq, maxval=-5, dtype=tf.int32, seed=42), dtype=tf.float32)
    mu_pos = tf.cast(tf.random.uniform(shape=(), minval=5, maxval=max_freq, dtype=tf.int32, seed=42), dtype=tf.float32)
    mu_stack = tf.stack([mu_neg, mu_pos], axis=-1)
    mu_choice = tf.squeeze(tf.random.categorical(logits=tf.math.log([[0.5, 0.5]]), num_samples=1))
    mu = mu_stack[mu_choice]

    sigma = tf.cast(tf.random.uniform(shape=(), minval=1, maxval=20, dtype=tf.int32, seed=seed), dtype=tf.float32)

    def offset(an_input):
        return tf.multiply(tf.convert_to_tensor(delta), tf.exp(
            tf.multiply(tf.constant(-0.5), tf.divide(
                tf.pow(tf.subtract(an_input, mu), tf.constant(2.)),
                tf.pow(sigma, tf.constant(2.))
            ))))

    if len(x.shape) == 4:
        interm_freq_offsets = offset(
            tf.reverse(tf.range(start=min_freq, limit=max_freq, delta=1, dtype=tf.float32), axis=[-1]))
        freq_offsets = interm_freq_offsets[tf.newaxis, :, tf.newaxis, tf.newaxis]
    else:
        interm_freq_offsets = offset(
            tf.reverse(tf.range(start=min_freq, limit=max_freq, delta=1, dtype=tf.float32), axis=[-1]))
        freq_offsets = interm_freq_offsets[:, tf.newaxis, tf.newaxis]

    freq_offsets = tf.broadcast_to(freq_offsets, x.shape)
    offset_mask = tf.subtract(tf.constant(1.), freq_offsets)

    return tf.multiply(x, offset_mask)


def sample_mixing(x, rate):
    x = tf.stack(x, axis=0)

    rate = tf.convert_to_tensor(rate, dtype=tf.float32)
    rate_update = tf.subtract(tf.constant(1., dtype=tf.float32), tf.gather(rate, [0]))
    rate = tf.tensor_scatter_nd_update(rate, tf.constant([[0]]), rate_update)

    if len(x.shape) == 5:
        rate = rate[:, tf.newaxis, tf.newaxis, tf.newaxis, tf.newaxis]
    else:
        rate = rate[:, tf.newaxis, tf.newaxis, tf.newaxis]

    rate = tf.broadcast_to(rate, x.shape)
    ret = tf.multiply(x, rate)

    return tf.reduce_sum(ret, axis=0)


def augment(x, is_fido=False):
    func_selection_nr = tf.random.uniform(shape=(), minval=0, maxval=7, seed=42, dtype=tf.int32)
    augmented_sample = tf.switch_case(branch_index=func_selection_nr, branch_fns={
        0: lambda: x[0],
        1: lambda: dropout(x[0], tf.random.uniform(shape=(), minval=0.0, maxval=0.11, dtype=tf.float32), 42),
        2: lambda: tf.keras.layers.GaussianNoise(
            tf.random.uniform(shape=(), minval=0.0, maxval=0.11, dtype=tf.float32))(x[0], training=True),
        3: lambda: time_stretch(x[0], tf.random.uniform(shape=(), minval=0.0, maxval=0.26, dtype=tf.float32)),
        4: lambda: spec_shift(x[0], tf.random.uniform(shape=(), minval=0, maxval=5, dtype=tf.int32)),
        5: lambda: scale(x[0], tf.random.uniform(shape=(), minval=0.0, maxval=0.26, dtype=tf.float32)),
        6: lambda: freq_filters(x[0], tf.random.uniform(shape=(), minval=0.0, maxval=0.08, dtype=tf.float32), 42, -60,
                                60),
        7: lambda: sample_mixing(x, tf.random.uniform(shape=(3,), minval=0.0, maxval=0.05, dtype=tf.float32))
    }, default=lambda: x[0])

    if is_fido:
        return tf.image.pad_to_bounding_box(augmented_sample, 4, 14, 128, 128), tf.image.pad_to_bounding_box(x[1], 4,
                                                                                                             14, 128,
                                                                                                             128)
    else:
        return tf.image.pad_to_bounding_box(augmented_sample, 4, 14, 128, 128)


def no_augment(x, is_fido=False):
    if is_fido:
        return tf.image.pad_to_bounding_box(x[0], 4, 14, 128, 128), tf.image.pad_to_bounding_box(x[1], 4, 14, 128, 128)
    else:
        return tf.image.pad_to_bounding_box(x[0], 4, 14, 128, 128)


def get_padding_function(dataset, datatype, is_fido=False):
    required_size = input_specs_dataset[dataset][datatype][:2]

    def pad_input(x):
        if is_fido:
            return tf.image.pad_to_bounding_box(x[0], 0, 0, *required_size), \
                   tf.image.pad_to_bounding_box(x[1], 0, 0, *required_size)
        else:
            return tf.image.pad_to_bounding_box(x[0], 0, 0, *required_size)

    return pad_input
