import keras_tuner
import numpy as np

from custom_items.data_fetching import get_dir_data, get_input_output_shape, dataset_constructor
from tuning_supervised import get_build_model_function


def run_model_on_data(model_name, dataset: str, datatype: str, backbone: str, experiment_type: str = None):
    print('testing model with', model_name, dataset, datatype, experiment_type)
    input_shape, output_shape, domain_shape = get_input_output_shape(dataset, datatype,
                                                                     experiment_type, return_domain=True)

    build_model = get_build_model_function(model_name, datatype, backbone, input_shape, output_shape,
                                           domain_shape)
    hp = keras_tuner.HyperParameters()
    model = build_model(hp)
    print(hp)
    print('model build')

    # padding to wanted size
    needs_domainclass = model_name == 'domain_class'
    is_fido = model_name == 'fido'

    path_dataset = get_dir_data(dataset, datatype, experiment_type)
    train_instances = np.array(list(range(1, 1000, 1)))
    train_set = dataset_constructor(train_instances, path_dataset, 'train', 12, is_fido,
                                    needs_domainclass)
    model.fit(x=train_set, epochs=1,
              steps_per_epoch=1,
              verbose=1)


def run_model_on_emptydata(model_name):
    print('testing model', model_name)
    input_shape, output_shape, domain_shape = (128, 2048, 6), 6, 150

    build_model = get_build_model_function(model_name, 'dfs', 'efficientnet', input_shape, output_shape,
                                           domain_shape)
    hp = keras_tuner.HyperParameters()
    model = build_model(hp)
    print(hp)
    print('model build')

    data = np.ones((1, *input_shape))
    print(data.shape)
    out = model(data)
    print(out)
    print(out[0].shape)


if __name__ == '__main__':
    # run_model_on_data('std', 'signfi', 'gaf', 'efficientnet', 'environment')
    # run_model_on_data('std', 'signfi', 'gaf', 'efficientnet', 'users')
    # run_model_on_data('std', 'signfi', 'dfs', 'efficientnet', 'environment')
    # run_model_on_data('std', 'signfi', 'gaf', 'vgg', 'environment')
    # run_model_on_data('std', 'signfi', 'gaf', 'resnet', 'environment')
    # run_model_on_data('std', 'signfi', 'dfs', 'vgg', 'environment')
    # run_model_on_data('std', 'signfi', 'dfs', 'resnet', 'environment')

    # run_model_on_data('domain_class', 'signfi', 'gaf', 'efficientnet', 'environment')
    # run_model_on_data('wigrunt', 'signfi', 'gaf', 'efficientnet', 'environment')
    # run_model_on_data('minibatch', 'signfi', 'gaf', 'efficientnet', 'environment')
    # run_model_on_data('fido', 'signfi', 'gaf', 'efficientnet', 'environment')

    # python cross_validation_experiment.py -e_s 1 -b_s 12 -d_f_n 'users' -cv_s 0 -g 0 -l_r '0.0001' -m_n 'std' -f_p "tmp/45105d13ccb9452d9fb7a45540f34564.pickle" -d signfi -d_t gaf -bb efficientnet
    run_model_on_emptydata('fido')
