import tensorflow as tf
import pandas as pd
import keras_tuner
import datetime
import pickle
import os

from models import domain_techniques, EfficientNetGafBackbone, EfficientNetDfsBackbone, BackboneResNet, BackboneVGG
from custom_items.data_fetching import dataset_constructor, get_dir_data, get_input_output_shape
from custom_items.tuners import get_best_tuner_hp, get_hyperband_tuner
from custom_items.data_fetching import fetch_labels_indices
from custom_items.args_parser import model_args_parser

# Set logging level
tf.get_logger().setLevel("WARNING")


def check_gpu_device(args):
    # GPU config. for allocating limited amount of memory on a given device
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            tf.config.experimental.set_memory_growth(gpus[args.gpu], True)
            tf.config.experimental.set_visible_devices(gpus[args.gpu], "GPU")
        except RuntimeError as e:
            print(e)


def create_fixed_hp(best_hps: keras_tuner.HyperParameters) -> keras_tuner.HyperParameters:
    new_hp = keras_tuner.HyperParameters()

    # ignore attention hps thus we add them to the tuning process
    attention_hps = ['kernel_size_conv_att_a_i', 'kernel_size_conv_att_a_j', 'strides_mlp_b_enc_i',
                     'strides_mlp_b_enc_j', 'pool_size_avg_enc_i', 'pool_size_avg_enc_j', 'pool_size_max_enc_i',
                     'pool_size_max_enc_j', 'classifier_depth', 'neurons classifier layer 0']

    for key, value in best_hps.values.items():
        if 'tuner/' in key or key in attention_hps:
            # skip tuner settings and attention layers for tuning
            continue
        new_hp.Fixed(key, value=value)

    return new_hp


def get_best_model(args):
    if args.backbone == 'efficientnet':
        if args.datatype == 'gaf':
            backbone = EfficientNetGafBackbone
        elif args.datatype == 'dfs':
            backbone = EfficientNetDfsBackbone
        else:
            raise ValueError("Unknown backbone name was encountered.")

    elif args.backbone == 'resnet':
        backbone = BackboneResNet

    elif args.backbone == 'vgg':
        backbone = BackboneVGG

    else:
        raise ValueError("Unknown backbone name was encountered.")

    intput_shape, output_shape, domain_shape = get_input_output_shape(args.dataset, args.datatype,
                                                                      args.domain_factor_name, return_domain=True)
    is_fido = False

    if args.model_name == 'std':
        build_model = domain_techniques.std.get_build_model(intput_shape, output_shape, backbone)
    elif args.model_name == 'fido':
        build_model = domain_techniques.fido.get_build_model(intput_shape, output_shape, backbone)
        is_fido = True
    elif args.model_name == 'wigrunt':
        build_model = domain_techniques.wigrunt.get_build_model(intput_shape, output_shape, backbone)
    elif args.model_name == 'domain_class':
        build_model = domain_techniques.domain_class.get_build_model(intput_shape, output_shape, domain_shape, backbone)
    elif args.model_name == 'minibatch':
        build_model = domain_techniques.minibatch.get_build_model(intput_shape, output_shape, backbone)
    else:
        raise ValueError("Unknown model name was encountered.")

    # reload hyperband tuning results and run experiment on best model to get results
    if args.model_name == 'wigrunt':
        best_hps = get_best_tuner_hp(build_model, 'std', args.datatype)
        fixed_hps = create_fixed_hp(best_hps)
        tuner = get_hyperband_tuner(build_model, args.model_name, args.datatype, hp=fixed_hps)

    else:
        tuner = get_hyperband_tuner(build_model, args.model_name, args.datatype)

    tuner.reload()

    best_hps = tuner.oracle.get_best_trials(num_trials=2000)
    best_kappa_values = [x.metrics.get_best_value('val_cohen_kappa') for x in best_hps]
    best_kappa_index = -max((x, -i) for i, x in enumerate(best_kappa_values))[1]
    model_hps_to_be_used = best_hps[best_kappa_index].hyperparameters

    best_model = build_model(model_hps_to_be_used)

    return best_model, is_fido


def load_data(is_fido):
    # Load training, validation, and test data
    with open(args.file_path, 'rb') as handle:
        indices_types_dict = pickle.load(handle)

    path_dataset = get_dir_data(args.dataset, args.datatype, args.signfi_experiment, local=False)
    train_instances, val_instances = fetch_labels_indices(f_path=path_dataset,
                                                          indices=indices_types_dict["train_indices"],
                                                          domain_types=indices_types_dict["train_types"])

    test_instances = indices_types_dict["test_indices"]
    os.remove(args.file_path)

    path_dataset = get_dir_data(args.dataset, args.datatype, args.signfi_experiment)
    train_set = dataset_constructor(train_instances, path_dataset, 'train', args.batch_size, is_fido)
    val_set = dataset_constructor(val_instances, path_dataset, 'val', args.batch_size, is_fido)
    test_set = dataset_constructor(test_instances, path_dataset, 'val', args.batch_size, is_fido)

    return train_set, val_set, test_set, train_instances, val_instances, test_instances


def run_experiment(best_model, train_instances, val_instances, test_instances, train_set, val_set, test_set, is_fido):
    callback_objects = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0, patience=5, restore_best_weights=True)
    ]

    train_history = best_model.fit(x=train_set, epochs=args.epoch_size,
                                   steps_per_epoch=len(train_instances) // args.batch_size,
                                   verbose=2, callbacks=callback_objects, validation_data=val_set,
                                   validation_steps=len(val_instances) // args.batch_size,
                                   validation_freq=1)
    history_frame = pd.DataFrame(train_history.history)
    
    output = best_model.evaluate(x=test_set, steps=len(test_instances) // args.batch_size, verbose=2)
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
                                 columns=['A', 'P', 'R', 'F', 'CK'])

    results_frame = pd.concat([history_frame, metrics_frame], axis=1)
    results_frame.to_csv(
        "results/{}_results_dfn_{}_cvs_{}_{}.csv".format(args.model_name, args.domain_factor_name,
                                                         str(args.crossval_split),
                                                         datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')))


if __name__ == '__main__':
    parser = model_args_parser()
    args = parser.parse_args()

    check_gpu_device(args)
    best_m, isfido = get_best_model(args)
    trs, vs, tes, tri, vi, tei = load_data(isfido)
    run_experiment(best_m, tri, vi, tei, trs, vs, tes, isfido)
