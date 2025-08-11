import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.math import generic as generic_math


# https://github.com/tensorflow/tensorflow/issues/36327
def keras_model_memory_usage_in_bytes(model, *, batch_size: int):
    """
    Return the estimated memory usage of a given Keras model in bytes.
    This includes the model weights and layers, but excludes the dataset.

    The model shapes are multipled by the batch size, but the weights are not.

    Args:
        model: A Keras model.
        batch_size: The batch size you intend to run the model with. If you
            have already specified the batch size in the model itself, then
            pass `1` as the argument here.
    Returns:
        An estimate of the Keras model's memory usage in bytes.

    """
    default_dtype = tf.keras.backend.floatx()
    shapes_mem_count = 0
    internal_model_mem_count = 0
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            internal_model_mem_count += keras_model_memory_usage_in_bytes(layer, batch_size=batch_size)
        single_layer_mem = tf.as_dtype(layer.dtype or default_dtype).size
        out_shape = layer.output_shape
        if isinstance(out_shape, list):
            out_shape = out_shape[0]
        for s in out_shape:
            if s is None:
                continue
            single_layer_mem *= s
        shapes_mem_count += single_layer_mem

    trainable_count = sum([tf.keras.backend.count_params(p) for p in model.trainable_weights])
    non_trainable_count = sum([tf.keras.backend.count_params(p) for p in model.non_trainable_weights])

    total_memory = (batch_size * shapes_mem_count + internal_model_mem_count + trainable_count + non_trainable_count)
    return round(total_memory * 1.15)  # To account for 10% discrepancy as indicated by author


# Assumes row-major flattening procedure for creation of domain label
# (see: https://eli.thegreenplace.net/2015/memory-layout-of-multi-dimensional-arrays)
def domain_to_be_left_out_indices_calculation(dim_nr, dim_ind, unflat_domain_label_shape):
    base_dim_range_lists = [list(range(x)) for x in unflat_domain_label_shape]
    base_dim_range_lists[dim_nr] = [dim_ind]

    test_indices = [
        w + unflat_domain_label_shape[-1] * x + unflat_domain_label_shape[-1] * unflat_domain_label_shape[-2] * y +
        unflat_domain_label_shape[-1] * unflat_domain_label_shape[-2] * unflat_domain_label_shape[-3] * z for w in
        base_dim_range_lists[-1] for x in base_dim_range_lists[-2] for y in base_dim_range_lists[-3] for z in
        base_dim_range_lists[-4]]

    # test_indices.sort()
    # print(test_indices)

    return test_indices


@tf.custom_gradient
def _stable_grad_softplus(x):
    """A (more) numerically stable softplus than `tf.nn.softplus`."""
    x = tf.convert_to_tensor(x)
    if x.dtype == tf.float64:
        cutoff = -20
    else:
        cutoff = -9

    y = tf.where(x < cutoff,
                 tf.math.log1p(
                     tf.clip_by_value(tf.exp(tf.clip_by_value(x, clip_value_min=-100000., clip_value_max=75.)),
                                      clip_value_min=-0.9999999, clip_value_max=100000.)),
                 tf.nn.softplus(x)
                 )

    def grad_fn(dy):
        return dy * tf.where(x < cutoff,
                             tf.exp(tf.clip_by_value(x, clip_value_min=-100000., clip_value_max=75.)),
                             tf.nn.sigmoid(x)
                             )

    return y, grad_fn


def _stable_sigmoid(x):
    """A (more) numerically stable sigmoid than `tf.math.sigmoid`."""
    x = tf.convert_to_tensor(x)
    if x.dtype == tf.float64:
        cutoff = -20
    else:
        cutoff = -9

    return tf.where(x < cutoff,
                    tf.exp(tf.clip_by_value(x, clip_value_min=-100000., clip_value_max=75.)),
                    tf.math.sigmoid(x)
                    )


class CorrectedSigmoid(tfp.bijectors.Sigmoid):
    def _forward(self, x):
        if self._is_standard_sigmoid:
            return _stable_sigmoid(x)
        lo = tf.convert_to_tensor(self.low)  # Concretize only once
        hi = tf.convert_to_tensor(self.high)
        diff = hi - lo
        left = lo + diff * _stable_sigmoid(x)
        right = hi - diff * _stable_sigmoid(-x)
        return tf.where(x < 0, left, right)

    def _inverse(self, y):
        y_clip_log = tf.clip_by_value(y, clip_value_min=1e-5, clip_value_max=100000.)
        y_clip_log1p = tf.clip_by_value(y, clip_value_min=-100000., clip_value_max=0.99999)

        if self._is_standard_sigmoid:
            return tf.math.log(y_clip_log) - tf.math.log1p(-y_clip_log1p)
        return tf.math.log(y_clip_log - self.low) - tf.math.log(self.high - y_clip_log)

    def _forward_log_det_jacobian(self, x):
        sigmoid_fldj = -_stable_grad_softplus(-x) - _stable_grad_softplus(
            x)  # pylint: disable=invalid-unary-operand-type
        if self._is_standard_sigmoid:
            return sigmoid_fldj
        return sigmoid_fldj + tf.math.log(self.high - self.low)


class CorrectedReciprocal(tfp.bijectors.Reciprocal):
    def _forward(self, x):
        x_non_zero = tf.where(x == 0., x + tf.constant(1e-5, shape=()), x)
        return 1. / x_non_zero

    def _forward_log_det_jacobian(self, x):
        x_non_zero = tf.where(x == 0., x + tf.constant(1e-5, shape=()), x)
        return -2. * tf.math.log(tf.math.abs(x_non_zero))


class CorrectedSoftplus(tfp.bijectors.Softplus):
    def _forward(self, x):
        if self.hinge_softness is None:
            y = _stable_grad_softplus(x)
        else:
            hinge_softness = tf.cast(self.hinge_softness, x.dtype)
            y = hinge_softness * _stable_grad_softplus(x / hinge_softness)

        # TF 2.5.0: return y + self.low if self.low is not None else y
        return y

    def _inverse(self, y):
        y_clip = tf.clip_by_value(y, clip_value_min=1e-5, clip_value_max=100000.)

        # TF 2.5.0: y_clip = y_clip - self.low if self.low is not None else y_clip
        if self.hinge_softness is None:
            return generic_math.softplus_inverse(y_clip)
        hinge_softness = tf.cast(self.hinge_softness, y_clip.dtype)
        return hinge_softness * generic_math.softplus_inverse(y_clip / hinge_softness)

    def _inverse_log_det_jacobian(self, y):
        # TF 2.5.0: y = y - self.low if self.low is not None else y
        if self.hinge_softness is not None:
            y = y / tf.cast(self.hinge_softness, y.dtype)
        return -tf.math.log(
            tf.clip_by_value(-tf.math.expm1(tf.clip_by_value(-y, clip_value_min=-100000., clip_value_max=75.)),
                             clip_value_min=1e-5, clip_value_max=100000.))


class CorrectedExp(tfp.bijectors.Exp):
    def _forward(self, x):
        return tf.exp(tf.clip_by_value(x, clip_value_min=-100000., clip_value_max=75.))

    def _inverse(self, y):
        return tf.math.log(tf.clip_by_value(y, clip_value_min=1e-5, clip_value_max=100000.))

    def _inverse_log_det_jacobian(self, y):
        power = tf.cast(self.power, y.dtype)
        return (power - tf.ones([], y.dtype)) * tf.math.log(
            tf.clip_by_value(y, clip_value_min=1e-5, clip_value_max=100000.))

    def _forward_log_det_jacobian(self, x):
        return x
