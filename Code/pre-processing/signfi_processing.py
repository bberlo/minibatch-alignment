import argparse
import os

import h5py
import numpy as np
import scipy.io as sio

from signfi_csi_dfs import convert_csi_to_dfs
from signfi_csi_gaf import convert_signfi_csi_data_to_gaf

input_dir_signfi = 'not-processed/'
root_dir_signfi = 'processed/'

# Signfi domain label is specified as 0-4 are users 1 to 5 from lab, and 5-9 are users 1 to 5 from home environment


def create_signfi_users_dataset(datatype: str):
    """
    Create a hdf5 file dataset of the signfi dataset with the different users
    :param datatype:
    :return:
    """
    print(f'Creating users dataset signfi, for {datatype}')
    # read dataset
    dataset_lab_150 = sio.loadmat(os.path.join(input_dir_signfi, 'dataset_lab_150'))
    print(f'Loaded in data')

    if datatype == 'gaf':
        output_file = os.path.join(root_dir_signfi, 'gaf', f'signfi-users-gaf.hdf5')
    elif datatype == 'dfs':
        output_file = os.path.join(root_dir_signfi, 'dfs', f'signfi-users-dfs.hdf5')
    else:
        raise ValueError(f'Unknown datatype {datatype}')
    f = h5py.File(output_file, 'w')

    nr_labels = dataset_lab_150['label'].astype('int8')
    # make one hot encoding for labels
    print('Max value labels: ', nr_labels.max())
    labels = np.zeros(shape=(nr_labels.shape[0], nr_labels.max()), dtype='int8')
    labels[np.arange(labels.shape[0]), nr_labels.squeeze() - 1] = 1
    # add label directory
    f.create_dataset('task_labels', data=labels, dtype='int8')

    user_objs = ['csi1', 'csi2', 'csi3', 'csi4', 'csi5']

    for i, obj in enumerate(user_objs):
        print(f'Running object {obj}')
        # preprocess dataset
        if datatype == 'gaf':
            csi_pre = convert_signfi_csi_data_to_gaf(dataset_lab_150[obj])
        elif datatype == 'dfs':
            csi_pre = convert_csi_to_dfs(dataset_lab_150[obj])
        else:
            raise ValueError(f'Unknown datatype {datatype}')

        # add dimension for channel dimension
        csi_pre = np.expand_dims(csi_pre, axis=-1)

        # add to datafile
        if i == 0:
            # Create the dataset at first
            f.create_dataset('inputs', data=csi_pre, dtype='float32', chunks=True, maxshape=(None, *csi_pre.shape[1:]))

            # add domain label
            domain_label = np.zeros(shape=(csi_pre.shape[0], 10), dtype='int8')
            domain_label[:, i] = 1
            f.create_dataset('domain_labels', data=domain_label, dtype='int8', chunks=True, maxshape=(None, 10))
        else:
            # Append new data to it
            f['inputs'].resize((f['inputs'].shape[0] + csi_pre.shape[0]), axis=0)
            f['inputs'][-csi_pre.shape[0]:] = csi_pre

            # add domain label
            # data is all from 5th person
            domain_label = np.zeros(shape=(csi_pre.shape[0], 10))
            domain_label[:, i] = 1
            f['domain_labels'].resize((f['domain_labels'].shape[0] + domain_label.shape[0]), axis=0)
            f['domain_labels'][-domain_label.shape[0]:] = domain_label

    print('done processing')
    f.close()


def create_signfi_environment_dataset(datatype: str):
    """
    Create a hdf5 file dataset of the signfi dataset with the different users
    :param datatype:
    :return:
    """
    print(f'Creating environment dataset signfi, for {datatype}')
    # read dataset
    dataset_lab_276_dl = sio.loadmat(os.path.join(input_dir_signfi, 'dataset_lab_276_dl'))
    dataset_home_276 = sio.loadmat(os.path.join(input_dir_signfi, 'dataset_home_276'))
    print(f'Loaded in data')

    if datatype == 'gaf':
        output_file = os.path.join(root_dir_signfi, 'gaf', f'signfi-environment-gaf.hdf5')
    elif datatype == 'dfs':
        output_file = os.path.join(root_dir_signfi, 'dfs', f'signfi-environment-dfs.hdf5')
    else:
        raise ValueError(f'Unknown datatype {datatype}')
    f = h5py.File(output_file, 'w')

    # add label directory
    label_lab = dataset_lab_276_dl['label_lab']
    label_home = dataset_home_276['label_home']
    labels_stacked = np.concatenate([label_lab, label_home], axis=0)
    # make one hot encoding for labels
    print('Max value labels: ', labels_stacked.max())
    labels = np.zeros(shape=(labels_stacked.shape[0], labels_stacked.max()), dtype='int8')
    labels[np.arange(labels.shape[0]), labels_stacked.squeeze() - 1] = 1
    f.create_dataset('task_labels', data=labels, dtype='int8')

    # preprocess dataset
    if datatype == 'gaf':
        csi_lab = convert_signfi_csi_data_to_gaf(dataset_lab_276_dl['csid_lab'])
        csi_home = convert_signfi_csi_data_to_gaf(dataset_home_276['csid_home'])
        csi_pre = np.concatenate([csi_lab, csi_home], axis=0)
    elif datatype == 'dfs':
        csi_lab = convert_csi_to_dfs(dataset_lab_276_dl['csid_lab'])
        csi_home = convert_csi_to_dfs(dataset_home_276['csid_home'])
        csi_pre = np.concatenate([csi_lab, csi_home], axis=0)
    else:
        raise ValueError(f'Unknown datatype {datatype}')

    # add dimension for channel dimension
    csi_pre = np.expand_dims(csi_pre, axis=-1)

    # Create the dataset at first
    f.create_dataset('inputs', data=csi_pre, dtype='float32', chunks=True, maxshape=(None, *csi_pre.shape[1:]))

    # add domain label
    domain_label = np.zeros(shape=(csi_pre.shape[0], 10))
    domain_label[:csi_lab.shape[0], 4] = 1  # lab 5th person
    domain_label[csi_lab.shape[0]:, 9] = 1  # home 5th person

    f.create_dataset('domain_labels', data=domain_label, dtype='int8', chunks=True, maxshape=(None, 10))

    print('done processing')
    f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert signfi CSI dataset to dfs or gaf')
    parser.add_argument('-dt', '--datatype', help='<Required> Which datatype to convert to',
                        required=True, choices=['dfs', 'gaf'])
    args = parser.parse_args()

    create_signfi_users_dataset(args.datatype)
    create_signfi_environment_dataset(args.datatype)
