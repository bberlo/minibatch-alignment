import math

import tensorflow as tf


# Callback to halt training when loss is negative or diverges (EarlyStopping doesn't account for this)
class HaltCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs.get('loss') < 0.0 or logs.get('val_loss') < 0.0 or math.isnan(logs.get('loss')) or math.isnan(
                logs.get('val_loss')):
            self.model.stop_training = True
