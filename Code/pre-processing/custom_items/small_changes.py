import os

import h5py
import numpy as np

from signfi_processing import root_dir_signfi

root_dir_signfi = '..\Datasets\SignFi'


def hd_labels_to_onehot(file: str):
    hd_file = h5py.File(file, 'r+')

    labels = hd_file['task_labels'][:]
    print(labels.shape)
    if labels.shape[1] == 1:
        print('Changing task labels to one hot')

        # make one hot encoding for labels
        print(labels.max())
        labels_onehot = np.zeros(shape=(labels.shape[0], labels.max()), dtype='int8')
        labels_onehot[np.arange(labels_onehot.shape[0]), labels.squeeze() - 1] = 1
        print(labels_onehot[:10])
        del hd_file['task_labels']
        hd_file['task_labels'] = labels_onehot
        print(hd_file['task_labels'].shape)
    else:
        print('not changing file')


def convert_all_signfi_onehot():
    files = [os.path.join(root_dir_signfi, 'gaf', f'signfi-environment-gaf.hdf5'),
             os.path.join(root_dir_signfi, 'dfs', f'signfi-environment-dfs.hdf5'),
             os.path.join(root_dir_signfi, 'gaf', f'signfi-users-gaf.hdf5'),
             os.path.join(root_dir_signfi, 'dfs', f'signfi-users-dfs.hdf5')]
    for file in files:
        hd_labels_to_onehot(file)


def add_dimension_input(file: str):
    print('changing file', file)
    hd_file = h5py.File(file, 'r+')

    data = hd_file['inputs'][:]
    print(data.shape)
    if data.shape[-1] != 1:
        data = np.expand_dims(data, axis=-1)
        print(data.shape)
        del hd_file['inputs']
        hd_file['inputs'] = data


def convert_all_signfi_datashape():
    files = [os.path.join(root_dir_signfi, 'gaf', f'signfi-environment-gaf.hdf5'),
             os.path.join(root_dir_signfi, 'dfs', f'signfi-environment-dfs.hdf5'),
             os.path.join(root_dir_signfi, 'gaf', f'signfi-users-gaf.hdf5'),
             os.path.join(root_dir_signfi, 'dfs', f'signfi-users-dfs.hdf5')
             ]
    for file in files:
        add_dimension_input(file)


if __name__ == '__main__':
    convert_all_signfi_datashape()
