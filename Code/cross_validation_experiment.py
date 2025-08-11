from datetime import datetime
import tensorflow as tf
from typing import Dict
import pandas as pd
import keras_tuner
import pickle
import math
import os

from custom_items.data_fetching import get_input_output_shape, get_dir_data, dataset_constructor, fetch_labels_indices
from tuning_supervised import get_build_model_function
from custom_items.args_parser import model_args_parser

# Set logging level
tf.get_logger().setLevel("WARNING")


def create_fixed_hp_dict(hps: Dict) -> keras_tuner.HyperParameters:
    """
    Create a fixed hp with the gives hps
    :param hps: hps to create fixed values for
    :return: a HP object
    """
    new_hp = keras_tuner.HyperParameters()

    for key, value in hps.items():
        new_hp.Fixed(key, value=value)

    return new_hp


def load_best_tuned_model(args):
    """
    Load the model with hp of best tuning results
    :return:
    """
    tuning_wigrunt_dfs = {
        'strides_mlp_b_enc_j': 22,
        'strides_mlp_b_enc_i': 18,
        'pool_size_avg_enc_i': 34,
        'pool_size_avg_enc_j': 18,
        'shared_mlp_depth': 4,
        'pool_size_max_enc_i': 38,
        'pool_size_max_enc_j': 14,
        'kernel_size_conv_att_a_i': 88,
        'kernel_size_conv_att_a_j': 56,
    }

    tuning_wigrunt_gaf = {
        'strides_mlp_b_enc_i': 22,
        'strides_mlp_b_enc_j': 26,
        'pool_size_avg_enc_i': 22,
        'pool_size_avg_enc_j': 20,
        'shared_mlp_depth': 4,
        'pool_size_max_enc_i': 28,
        'pool_size_max_enc_j': 14,
        'kernel_size_conv_att_a_i': 48,
        'kernel_size_conv_att_a_j': 32,
    }

    input_shape, output_shape, domain_shape = get_input_output_shape(args.dataset, args.datatype,
                                                                     args.domain_factor_name, return_domain=True)

    build_model = get_build_model_function(args.model_name, args.datatype, args.backbone, input_shape, output_shape,
                                           domain_shape)

    if args.model_name == 'wigrunt':
        hp_fixed = create_fixed_hp_dict(tuning_wigrunt_dfs if args.datatype == 'dfs' else tuning_wigrunt_gaf)
        # set best tuning results of wigrunt parameters, for the rest use default which is the tuned backbone
        best_model = build_model(hp=hp_fixed)
    else:
        # use default parameters as that are the tuned parameters
        best_model = build_model(hp=keras_tuner.HyperParameters())

    return best_model


def load_data(args):
    """
    Load the data for the specific indices provided in the pickle by the main program
    :param args:
    :return:
    """
    # Load training, validation, and test data
    with open(args.file_path, 'rb') as handle:
        indices_types_dict = pickle.load(handle)

    path_dataset = get_dir_data(args.dataset, args.datatype, args.domain_factor_name, local=False)
    train_instances, val_instances = fetch_labels_indices(f_path=path_dataset,
                                                          indices=indices_types_dict["train_indices"],
                                                          domain_types=indices_types_dict["train_types"])
    test_instances = indices_types_dict["test_indices"]
    os.remove(args.file_path)

    needs_domainclass = args.model_name == 'domain_class'
    is_fido = args.model_name == 'fido'

    train_set = dataset_constructor(train_instances, path_dataset, 'train', args.batch_size, is_fido,
                                    needs_domainclass)
    val_set = dataset_constructor(val_instances, path_dataset, 'val', args.batch_size, is_fido,
                                  needs_domainclass)
    test_set = dataset_constructor(test_instances, path_dataset, 'val', args.batch_size, is_fido,
                                   needs_domainclass)

    return train_instances, val_instances, test_instances, train_set, val_set, test_set


def train_model_and_test(args, model):
    """
    Train a model with given instances of the data from a certain domain
    and report back results
    :param args: arguments
    :param model: the model to train
    :return:
    """

    # Callback to halt training when loss is negative or diverges (EarlyStopping doesn't account for this)
    class HaltCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if logs.get('loss') < 0.0 or logs.get('val_loss') < 0.0 or math.isnan(logs.get('loss')) \
                    or math.isnan(logs.get('val_loss')) or math.isnan(logs.get('bilinear_similarity_loss')) \
                    or logs.get('loss') > 100000.0 or logs.get('val_loss') > 100000.0:
                self.model.stop_training = True
                logs['val_ntxent_sim_loss'] = 1000.0
                logs['val_domain_class_loss'] = 0.0

    callback_objects = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0, patience=12, restore_best_weights=True),
        # HaltCallback()
    ]
    # get data
    train_instances, val_instances, test_instances, train_set, val_set, test_set = load_data(args)

    # fit model
    train_history = model.fit(x=train_set, epochs=args.epoch_size,
                              steps_per_epoch=len(train_instances) // args.batch_size,
                              verbose=2, callbacks=callback_objects, validation_data=val_set,
                              validation_steps=len(val_instances) // args.batch_size,
                              validation_freq=1)
    history_frame = pd.DataFrame(train_history.history)

    # perform experiment and write results
    output = model.evaluate(x=test_set, steps=len(test_instances) // args.batch_size, verbose=2)
    if args.model_name == 'fido':
        acc, precision, recall, f_score, kappa_score, _, _, _ = output
    elif args.model_name == 'minibatch':
        # minibatch has loss - task_output_loss - bilinear_similarity_loss - task_output_categorical_accuracy - 
        # task_output_precision - task_output_recall - task_output_f1_score - task_output_cohen_kappa
        _, _, _, acc, precision, recall, f_score, kappa_score = output
    elif args.model_name == 'domain_class':
        _, _, _, acc, precision, recall, f_score, kappa_score = output
    else:
        _, acc, precision, recall, f_score, kappa_score = output
    metrics_frame = pd.DataFrame(data=[[acc, precision, recall, f_score, kappa_score]],
                                 columns=['accuracy', 'precision', 'recall', 'f1_score', 'cohen_kappa'])

    os.makedirs(f'results/{args.dataset}_{args.datatype}/{args.model_name}', exist_ok=True)
    metrics_frame['splits_leftout'] = args.splits_leftout

    cur_date = datetime.now().strftime('%Y-%m-%d_%H-%M')
    metrics_frame.to_csv(
        f"results/{args.dataset}_{args.datatype}/{args.model_name}/{args.backbone}_evaluation_dfn_{args.domain_factor_name}_cvs_{args.crossval_split}_{cur_date}.csv",
        index=False)

    history_frame.to_csv(
        f"results/{args.dataset}_{args.datatype}/{args.model_name}/{args.backbone}_train_history_dfn_{args.domain_factor_name}_cvs_{args.crossval_split}_{cur_date}.csv",
        index=False)


def main():
    """
    Main entry to run cross validation experiments
    :return:
    """
    parser = model_args_parser()
    args = parser.parse_args()

    # load model
    best_model = load_best_tuned_model(args)
    if args.model_name != 'fido':
        tf.keras.backend.set_value(best_model.optimizer.learning_rate, args.learning_rate)

    # run experiment
    train_model_and_test(args, best_model)


if __name__ == '__main__':
    main()
