import os
import time
from typing import List, Tuple

import h5py
import numpy as np
import scipy.io as sio
from pyts.image import GramianAngularField
from tqdm import tqdm

input_dir_signfi = 'not-processed/'
root_dir_signfi = 'processed/'

files_signfi = [('dataset_lab_276_ul', ['csiu_lab'], 'label_lab'),
                ('dataset_lab_276_dl', ['csid_lab'], 'label_lab'),
                ('dataset_home_276', ['csiu_home', 'csid_home'], 'label_home'),
                ('dataset_lab_150', ['csi1', 'csi2', 'csi3', 'csi4', 'csi5'], 'label')]


# =======================================================================================
# SignFi gaf part
# =======================================================================================

def signfi_csi_to_gaf(csi_matrix: np.ndarray, gadf) -> np.ndarray:
    """
    Convert a single signfi csi sample to gaf representation
    :param csi_matrix: the numpy csi matrix
    :param gadf: the gaf object to use for conversion
    :return: numpy array for
    """
    csi_matrix = csi_matrix.T

    # PCA analysis https://stackoverflow.com/questions/60508233/python-implement-a-pca-using-svd
    conj_multiplication_1 = csi_matrix.copy()
    conj_multiplication_1 -= np.mean(conj_multiplication_1, axis=0)
    U, S, Vt = np.linalg.svd(conj_multiplication_1, full_matrices=True)
    V = np.transpose(Vt)

    # Enforce sign convention https://nl.mathworks.com/matlabcentral/answers/300899-sign-difference-between-coeff-pca-x-and-v-svd-x
    max_indices = np.argmax(np.absolute(V), axis=0)
    max_value_signs = np.squeeze(np.sign(np.take_along_axis(V, np.expand_dims(max_indices, axis=0), axis=0)))
    max_value_signs = np.stack([max_value_signs] * V.shape[0])
    V = np.multiply(V, max_value_signs)

    conj_multiplication_pca = np.matmul(csi_matrix, V[:, 0])

    conj_multiplication_pca = conj_multiplication_pca.reshape(1, -1)

    # convert to abs
    conj_multiplication_pca = np.abs(conj_multiplication_pca)

    return gadf.fit_transform(conj_multiplication_pca)


def convert_signfi_csi_data_to_gaf(csi_matrix: np.ndarray) -> np.ndarray:
    """
    Convert csi data set to gaf representation
    :param csi_matrix: numpy array of csi data
    :return: numpy array of csi gaf
    """
    print('converting signfi to gaf')
    start = time.perf_counter()

    gadf = GramianAngularField(method='difference')

    csi_matrix = csi_matrix.T
    # stack antenna and subcarrier index
    csi_matrix = csi_matrix.reshape((csi_matrix.shape[0], -1, csi_matrix.shape[-1]))

    # preprocess dataset
    shape_gaf = (csi_matrix.shape[0], csi_matrix.shape[-1], csi_matrix.shape[-1])
    csi_gaf = np.empty(shape_gaf, dtype='float32')

    for index, res in enumerate(csi_matrix):
        csi_gaf[index] = signfi_csi_to_gaf(csi_matrix=res, gadf=gadf)

    print('Processing csi to gaf took: ', time.perf_counter() - start)

    return csi_gaf


def signfi_create_gaf_dataset(data_file: str, csi_obj: List[str], gaf_options: Tuple[str, str], label_obj: str):
    # read dataset
    data_lab_276_ul = sio.loadmat(os.path.join(input_dir_signfi, data_file))
    print(f'Loaded in {data_file}')
    print(f'With gaf options {gaf_options}')

    output_file = os.path.join(root_dir_signfi, 'gaf', f'{data_file}-gaf.hdf5')
    f = h5py.File(output_file, 'w')
    gadf = GramianAngularField(method=gaf_options[1])

    labels = data_lab_276_ul[label_obj]
    # add label directory
    f.create_dataset('label', data=labels, dtype='int8')

    for obj in csi_obj:
        print(f'Running object {obj}')
        csi = data_lab_276_ul[obj].T
        # stack antenna and subcarrier index
        csi = csi.reshape((csi.shape[0], -1, csi.shape[-1]))

        # preprocess dataset
        shape_gaf = (csi.shape[0], csi.shape[-1], csi.shape[-1])
        dset = f.create_dataset(f"{obj}_{gaf_options[0]}_{gaf_options[1][:3]}", shape_gaf)

        for index, res in tqdm(enumerate(csi)):
            dset[index] = signfi_csi_to_gaf(csi_matrix=csi[index], gadf=gadf)

    print('done processing')
    f.close()
