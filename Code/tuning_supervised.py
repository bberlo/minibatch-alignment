from kerastuner_tensorboard_logger import setup_tb
from multiprocess import set_start_method
import tensorflow as tf
import keras_tuner
import pickle
import os

from models import domain_techniques, EfficientNetGafBackbone, EfficientNetDfsBackbone, BackboneResNet, BackboneVGG
from custom_items.data_fetching import fetch_labels_indices, get_dir_data, get_input_output_shape
from custom_items.tuners import get_best_tuner_hp, get_hyperband_tuner
from custom_items.args_parser import model_args_parser

# Set logging level
tf.get_logger().setLevel("WARNING")


def disable_gpu():
    # Prevent main process from clogging up GPU memory
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            tf.config.set_visible_devices([], "GPU")
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


def get_build_model_function(model_name, datatype, backbone_name, inp_shape, out_shape, dom_shape):
    if backbone_name == 'efficientnet':
        if datatype == 'gaf':
            backbone = EfficientNetGafBackbone
        elif datatype == 'dfs':
            backbone = EfficientNetDfsBackbone
        else:
            raise ValueError("Unknown backbone name was encountered.")

    elif backbone_name == 'resnet':
        backbone = BackboneResNet

    elif backbone_name == 'vgg':
        backbone = BackboneVGG

    else:
        raise ValueError("Unknown backbone name was encountered.")

    if model_name == 'std':
        model_func = domain_techniques.std.get_build_model(inp_shape, out_shape, backbone)
    elif model_name == 'fido':
        model_func = domain_techniques.fido.get_build_model(inp_shape, out_shape, backbone)
    elif model_name == 'wigrunt':
        model_func = domain_techniques.wigrunt.get_build_model(inp_shape, out_shape, backbone)
    elif model_name == 'domain_class':
        model_func = domain_techniques.domain_class.get_build_model(inp_shape, out_shape, dom_shape, backbone)
    elif model_name == 'minibatch':
        model_func = domain_techniques.minibatch.get_build_model(inp_shape, out_shape, backbone)
    else:
        raise ValueError("Unknown model name was encountered.")

    return model_func


def main(args):
    # Prevent main process from clogging up GPU memory
    disable_gpu()

    # Load training, validation, and test data
    with open(args.file_path, 'rb') as handle:
        indices_types_dict = pickle.load(handle)

    path_dataset = get_dir_data(args.dataset, args.datatype, args.domain_factor_name, False)
    train_indices, val_indices = fetch_labels_indices(f_path=path_dataset,
                                                      indices=indices_types_dict["train_indices"],
                                                      domain_types=indices_types_dict["train_types"])

    # During tuning, test data is not considered. Added for code completion purposes.
    test_instances = indices_types_dict["test_indices"]

    os.remove(args.file_path)
    set_start_method("spawn")

    input_shape, output_shape, domain_shape = get_input_output_shape(args.dataset, args.datatype,
                                                                     args.domain_factor_name, return_domain=True)

    build_model = get_build_model_function(args.model_name, args.datatype, args.backbone, input_shape, output_shape,
                                           domain_shape)

    if args.model_name == 'wigrunt':
        best_hps = get_best_tuner_hp(build_model, 'std', args.datatype)
        fixed_hps = create_fixed_hp(best_hps)
        tuner = get_hyperband_tuner(build_model, args.model_name, args.datatype, hp=fixed_hps)

    else:
        tuner = get_hyperband_tuner(build_model, args.model_name, args.datatype)

    setup_tb(tuner)

    tuner.search(x=train_indices, epochs=args.epoch_size, steps_per_epoch=len(train_indices) // args.batch_size,
                 verbose=2, is_fido=args.model_name == 'fido', gpu=args.gpu,
                 validation_data=val_indices, validation_steps=len(val_indices) // args.batch_size,
                 validation_freq=1, batch_size=args.batch_size,
                 dataset_filepath=path_dataset)


if __name__ == '__main__':
    parser = model_args_parser()
    args = parser.parse_args()
    main(args)
