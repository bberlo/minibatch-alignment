import math
from typing import Callable

import keras_tuner
import keras_tuner as kt
import keras_tuner.engine.trial as trial_module
import tensorflow as tf
from kerastuner_tensorboard_logger import TensorBoardLogger
from multiprocess import Queue, Process

from custom_items.callbacks import HaltCallback
from custom_items.data_fetching import dataset_constructor, get_input_output_shape
from custom_items.utilities import keras_model_memory_usage_in_bytes


# To access best hyperparameters and train from scratch see https://github.com/keras-team/keras-tuner/issues/41
class HyperbandSizeFiltering(kt.tuners.hyperband.Hyperband):
    def __init__(self, hypermodel, objective, max_epochs, factor, hyperband_iterations,
                 seed, directory, project_name, logger, distribution_strategy=None, hyperparameters=None,
                 tune_new_entries=True):

        self.queue = Queue()
        self.max_model_size_in_bytes = 6000000000

        super(HyperbandSizeFiltering, self).__init__(
            hypermodel=hypermodel,
            objective=objective,
            hyperparameters=hyperparameters,
            tune_new_entries=tune_new_entries,
            max_epochs=max_epochs,
            factor=factor,
            hyperband_iterations=hyperband_iterations,
            seed=seed,
            distribution_strategy=distribution_strategy,
            directory=directory,
            project_name=project_name,
            logger=logger
        )

    def on_trial_end(self, trial):
        """A hook called after each trial is run.
        # Arguments:
            trial: A `Trial` instance.
        """
        # Send status to Logger
        if self.logger:
            self.logger.report_trial_state(trial.trial_id, trial.get_state())

        if not trial.get_state().get("status") == trial_module.TrialStatus.INVALID:
            self.oracle.end_trial(trial.trial_id, trial_module.TrialStatus.COMPLETED)

        self.oracle.update_space(trial.hyperparameters)
        # Display needs the updated trial scored by the Oracle.
        self._display.on_trial_end(self.oracle.get_trial(trial.trial_id))
        self.save()

    def _build_and_fit_model(self, trial, *fit_args, **fit_kwargs):
        p = Process(target=self._build_and_fit_model_worker,
                    args=(self.hypermodel, fit_args, fit_kwargs, trial.hyperparameters,
                          self.queue, self.max_model_size_in_bytes))
        p.start()
        ret = self.queue.get()
        p.join()

        if isinstance(ret, str):
            self.oracle.end_trial(trial.trial_id, trial_module.TrialStatus.INVALID)

            dummy_history_obj = tf.keras.callbacks.History()
            dummy_history_obj.history.setdefault('val_loss', []).append(2.5)
            return dummy_history_obj
        else:
            history_obj = tf.keras.callbacks.History()
            history_obj.history = ret
            return history_obj

    @staticmethod
    def _build_and_fit_model_worker(hypermodel, fit_args, fit_kwargs, hyperparams, queue, max_bytes):
        # OP placement analysis for discovering throughput bottlenecks
        # tf.debugging.set_log_device_placement(True)

        # GPU config. for allocating limited amount of memory on a given device
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            try:
                tf.config.experimental.set_memory_growth(gpus[fit_kwargs["gpu"]], True)
                tf.config.experimental.set_visible_devices(gpus[fit_kwargs["gpu"]], "GPU")
            except RuntimeError as e:
                print(e)

        # Set logging level
        tf.get_logger().setLevel("WARNING")

        fit_kwargs["callbacks"].extend([
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0, patience=5, restore_best_weights=True),
            HaltCallback()
        ])

        fit_kwargs["x"] = dataset_constructor(fit_kwargs["x"], fit_kwargs["dataset_filepath"],
                                              'train', fit_kwargs["batch_size"], fit_kwargs["is_fido"])
        fit_kwargs["validation_data"] = dataset_constructor(fit_kwargs["validation_data"],
                                                            fit_kwargs["dataset_filepath"],
                                                            'val', fit_kwargs["batch_size"], fit_kwargs["is_fido"])

        model = hypermodel.build(hyperparams)
        model_size_in_bytes = keras_model_memory_usage_in_bytes(model=model, batch_size=fit_kwargs["batch_size"])
        print("Considering model with size: {}".format(model_size_in_bytes))

        del fit_kwargs["dataset_filepath"]
        del fit_kwargs["batch_size"]
        del fit_kwargs["is_fido"]
        del fit_kwargs["gpu"]

        if model_size_in_bytes > max_bytes:
            queue.put('invalid')
        else:
            try:
                history = model.fit(*fit_args, **fit_kwargs)
                if history.history["loss"][-1] < 0.0 or history.history["val_loss"][-1] < 0.0 \
                        or math.isnan(history.history["loss"][-1]) or math.isnan(history.history["val_loss"][-1]):
                    queue.put('invalid')
                else:
                    queue.put(history.history)
            except (tf.errors.ResourceExhaustedError, tf.errors.InternalError):
                queue.put('invalid')


def get_hyperband_tuner(build_model: Callable, model_name: str, name_experiment: str,
                        hp=None) -> HyperbandSizeFiltering:
    """
    Setup the hyperband tuner
    :param build_model: function to get hp model
    :param hp: optional hyperparameters to provide the tuner
    :param name_experiment: name of expeirment
    :param model_name: name of model
    :return: the setup tuner
    """
    dir_output = f'Streams/{name_experiment}'

    tuner = HyperbandSizeFiltering(
        hypermodel=build_model,
        hyperparameters=hp,
        objective='val_loss',
        max_epochs=1000,
        factor=3,
        hyperband_iterations=1,
        seed=42,
        directory=dir_output,
        project_name=model_name,
        logger=TensorBoardLogger(metrics=["val_loss"], logdir=f'{dir_output}/{model_name}-hparams')
    )

    return tuner


def get_best_tuner_hp(build_model: Callable, model_name: str, name_experiment: str,
                      hp=None) -> keras_tuner.HyperParameters:
    """
    Reload the tuner and return the best hp results
    :param build_model: function to get hp model
    :param name_experiment: name experiment
    :param model_name: name of model
    :param hp: optional hyperparameters to provide the tuner
    :return: hyperparameters with best results
    """
    tuner = get_hyperband_tuner(build_model, model_name, name_experiment, hp=hp)
    tuner.reload()
    best_hps = tuner.get_best_hyperparameters()[0]

    return best_hps


def get_best_tuner_model(build_model: Callable, model_name: str, name_experiment: str, hp=None):
    """
    Reload the tuner and return the best hp results
    :param build_model: function to get hp model
    :param name_experiment: name experiment
    :param model_name: name of model
    :param hp: optional hyperparameters to provide the tuner
    :return: hyperparameters with best results
    """
    tuner = get_hyperband_tuner(build_model, model_name, name_experiment, hp=hp)
    tuner.reload()
    best_model = tuner.get_best_models()[0]

    return best_model
