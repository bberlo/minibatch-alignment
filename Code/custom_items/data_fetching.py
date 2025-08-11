from typing import Tuple, Union

import h5py
import numpy as np
import sklearn.model_selection as sk
import tensorflow as tf
import tensorflow_io as tfio

from custom_items.augmenters import get_padding_function
from custom_items.data_specs import input_specs_dataset, output_specs_dataset, data_paths, BASE_DATA_PATH_LOCAL, \
    BASE_DATA_PATH, domain_size_datasets


def get_input_output_shape(dataset: str, datatype: str, experiment_type: str = None, return_domain: bool = False) -> \
        Union[Tuple[Tuple[int], int], Tuple[Tuple[int], int, int]]:
    """
    Returns the input and output shape of the dataset
    :param return_domain: if to retun domain shape
    :param experiment_type: experiment type for signfi, data split of users or environment
    :param dataset: dataset to return either widar or signfi
    :param datatype: datatype to return either gaf or dfs
    :return:
    """
    assert dataset in ['signfi', 'widar'], f'unknown dataset {dataset}'
    assert datatype in ['dfs', 'gaf'], f'Unknown datatype {datatype}'
    if dataset == 'signfi':
        if experiment_type == 'random':
            # use environment if none specified
            experiment_type = 'environment'
        assert experiment_type in ['environment',
                                   'user'], f'Unknown experiment type for domain leave out, {experiment_type}'

    input_shape = input_specs_dataset[dataset][datatype]
    if dataset == 'widar':
        output_shape = output_specs_dataset[dataset]
    else:
        output_shape = output_specs_dataset[dataset][experiment_type]

    if return_domain:
        return input_shape, output_shape, domain_size_datasets[dataset]
    else:
        return input_shape, output_shape


def get_dir_data(dataset: str, datatype: str, experiment_type: str = None, local: bool = True) -> str:
    """
    returns directory of data
    :param local: if the dataset is on the local folder
    :param experiment_type: experiment type for signfi, data split of users or environment
    :param dataset: dataset to return either widar or signfi
    :param datatype: datatype to return either gaf or dfs
    :return: str of path to hd5 file
    """
    assert dataset in ['signfi', 'widar'], f'unknown dataset {dataset}'
    assert datatype in ['dfs', 'gaf'], f'Unknown datatype {datatype}'
    assert experiment_type in ['random', 'user', 'position', 'orientation',
                               'environment'], f'experiment type for domain leave out, {experiment_type}'
    if experiment_type == 'random':
        # use environment if none specified
        experiment_type = 'environment'

    if dataset == 'signfi':
        path_dataset = data_paths[dataset][datatype][experiment_type]
    elif dataset == 'widar':
        path_dataset = data_paths[dataset][datatype]
    else:
        raise ValueError(f'Unknown dataset {dataset}')

    if local:
        path_dataset = BASE_DATA_PATH_LOCAL + path_dataset
    else:
        path_dataset = BASE_DATA_PATH + path_dataset

    return path_dataset


def fetch_labels_indices(f_path, indices=None, domain_types=None):
    if indices is None:
        with h5py.File(f_path, 'r') as f:
            dset_1 = f['domain_labels']
            domain_labels = dset_1[:]

        return domain_labels

    else:
        with h5py.File(f_path, 'r') as f:
            all_domain_labels = f['domain_labels'][:]
            all_task_labels = f['task_labels'][:]

        domain_labels = all_domain_labels[indices]
        sparse_domain_labels = np.argmax(domain_labels, axis=1) + 1
        domain_types = np.asarray(domain_types)

        task_labels = all_task_labels[indices]
        sparse_task_labels = np.argmax(task_labels, axis=1)

        if 'signfi' in f_path:
            # signfi create validation split on task labels, because not enough domains
            k_fold_object = sk.StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
            train_samples, val_samples = next(k_fold_object.split(task_labels, sparse_task_labels))
            train_indices, val_indices = indices[train_samples], indices[val_samples]
        else:
            k_fold_object = sk.KFold(n_splits=10, shuffle=True, random_state=42)
            train_type_indices, val_type_indices = next(k_fold_object.split(X=np.expand_dims(a=domain_types, axis=1)))

            train_types, val_types = domain_types[train_type_indices], domain_types[val_type_indices]

            train_indices, val_indices = \
                np.where(np.isin(sparse_domain_labels, test_elements=train_types))[0], \
                np.where(np.isin(sparse_domain_labels, test_elements=val_types))[0]

        return train_indices, val_indices


def dataset_constructor(instances, f_path, subset_type, batch_size, is_fido=True, domain_class=False):
    dataset = 'widar' if 'widar' in f_path else 'signfi'
    datatype = 'gaf' if 'gaf.hdf5' in f_path else 'dfs'
    experiment_type = 'user' if 'user' in f_path else 'environment'
    input_shape, output_shape = get_input_output_shape(dataset, datatype, experiment_type)
    domain_shape = domain_size_datasets[dataset]

    widar_hdf5 = tfio.IOTensor.from_hdf5(
        filename=f_path,
        spec={'/inputs': tf.TensorSpec(shape=input_shape, dtype=tf.float32),
              '/task_labels': tf.TensorSpec(shape=output_shape, dtype=tf.int8),
              '/domain_labels': tf.TensorSpec(shape=domain_shape, dtype=tf.int8)})
    widar_inputs = widar_hdf5('/inputs')
    widar_task_labels = widar_hdf5('/task_labels')
    widar_domain_labels = widar_hdf5('/domain_labels')

    def get_sample(instance, label_type):
        if label_type == 'none':
            return widar_inputs[instance]
        elif label_type == 'task_label':
            return widar_inputs[instance], widar_task_labels[instance]
        elif label_type == 'task_domain_label':
            return widar_inputs[instance], widar_task_labels[instance], widar_domain_labels[instance]
        else:
            raise ValueError("Unknown label type definition.")

    map_func = get_padding_function(dataset, datatype, is_fido)

    if subset_type == 'train':
        individual_sets = [tf.data.Dataset.from_tensor_slices(instances).shuffle(buffer_size=len(instances),
                                                                                 reshuffle_each_iteration=True).repeat()
                           for _ in range(3)]
    else:
        individual_sets = [tf.data.Dataset.from_tensor_slices(instances).repeat() for _ in range(3)]

    if domain_class:
        individual_sets[0] = individual_sets[0].map(lambda x: get_sample(x, 'task_domain_label'))
        individual_sets[0] = individual_sets[0].map(lambda x, y, z: (map_func([x]), (y, z)))

        individual_sets[0] = individual_sets[0].batch(batch_size, True).prefetch(20)

        return individual_sets[0]
    else:
        individual_sets[0] = individual_sets[0].map(lambda x: get_sample(x, 'task_label'))
        individual_sets[1:] = [x.map(lambda y: get_sample(y, 'none')) for x in individual_sets[1:]]

        dset = tf.data.Dataset.zip(tuple(individual_sets))
        dset = dset.map(lambda x1y, x2, x3: (map_func([x1y[0], x2, x3]), x1y[1])).batch(batch_size, True).prefetch(20)
        return dset
