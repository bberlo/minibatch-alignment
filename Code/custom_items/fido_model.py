import tensorflow as tf


class FiDoModel(tf.keras.Model):

    def compile(self, optimizer='rmsprop', loss=None, class_loss_fn=None, reconst_loss_fn=None, metrics=None,
                loss_weights=None, weighted_metrics=None, run_eagerly=None, steps_per_execution=None, **kwargs):
        super(FiDoModel, self).compile(optimizer, loss, metrics, loss_weights, weighted_metrics,
                                       run_eagerly, steps_per_execution, **kwargs)

        self.class_loss_fn = class_loss_fn
        self.reconst_loss_fn = reconst_loss_fn

    def train_step(self, data):
        x, y, sample_weight = tf.keras.utils.unpack_x_y_sample_weight(data)  # x_label = x[0], x_unlabel = x[1]

        # Training the classification path
        with tf.GradientTape() as class_tape:
            y_pred, _ = self(x[0], training=True)
            class_loss = self.class_loss_fn(y, y_pred)
            if self.losses:
                class_loss += tf.add_n(self.losses)

        trainable_vars = self.trainable_variables
        class_gradients = class_tape.gradient(class_loss, trainable_vars)
        self.optimizer[0].apply_gradients(zip(class_gradients, trainable_vars))
        self.compiled_metrics.update_state(y, y_pred)

        # Training the reconstruction path
        with tf.GradientTape() as reconst_tape:
            _, x_pred = self(x[0], training=True)
            _, x_u_pred = self(x[1], training=True)

            reconst_loss = self.reconst_loss_fn(x[0], x_pred)
            reconst_u_loss = self.reconst_loss_fn(x[1], x_u_pred)
            overall_reconst_loss = reconst_loss + reconst_u_loss
            if self.losses:
                overall_reconst_loss += tf.add_n(self.losses)

        reconst_gradients = reconst_tape.gradient(overall_reconst_loss, trainable_vars)
        self.optimizer[1].apply_gradients(zip(reconst_gradients, trainable_vars))

        return_metrics = {}
        for metric in self.metrics:
            result = metric.result()
            if isinstance(result, dict):
                return_metrics.update(result)
            else:
                return_metrics[metric.name] = result
        return_metrics["loss"] = class_loss
        return_metrics["class_loss"] = class_loss
        return_metrics["reconst_loss"] = overall_reconst_loss

        return return_metrics

    def test_step(self, data):
        x, y, sample_weight = tf.keras.utils.unpack_x_y_sample_weight(data)  # x_label = x[0], x_unlabel = x[1]

        # Evaluate the classification path
        y_pred, _ = self(x[0], training=False)
        class_loss = self.class_loss_fn(y, y_pred)
        if self.losses:
            class_loss += tf.add_n(self.losses)
        self.compiled_metrics.update_state(y, y_pred)

        # Evaluate the reconstruction path
        _, x_pred = self(x[0], training=False)
        _, x_u_pred = self(x[1], training=False)

        reconst_loss = self.reconst_loss_fn(x[0], x_pred)
        reconst_u_loss = self.reconst_loss_fn(x[1], x_u_pred)
        overall_reconst_loss = reconst_loss + reconst_u_loss
        if self.losses:
            overall_reconst_loss += tf.add_n(self.losses)

        return_metrics = {}
        for metric in self.metrics:
            result = metric.result()
            if isinstance(result, dict):
                return_metrics.update(result)
            else:
                return_metrics[metric.name] = result
        return_metrics["loss"] = class_loss
        return_metrics["class_loss"] = class_loss
        return_metrics["reconst_loss"] = overall_reconst_loss

        return return_metrics
