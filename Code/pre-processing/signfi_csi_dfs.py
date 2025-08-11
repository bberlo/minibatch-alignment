import os
import time
from typing import List

import h5py
import numpy as np
import scipy.io as sio
from scipy import signal

input_dir_signfi = 'not-processed/'
root_dir_signfi = 'processed/'

files_signfi = [('dataset_lab_276_ul', ['csiu_lab'], 'label_lab'),
                ('dataset_lab_276_dl', ['csid_lab'], 'label_lab'),
                ('dataset_home_276', ['csiu_home', 'csid_home'], 'label_home'),
                ('dataset_lab_150', ['csi1', 'csi2', 'csi3', 'csi4', 'csi5'], 'label')]


def process_rx_sample(csi_matrix):
    """
    Process 1 csi sample and create the doppler frequency spectrum
    Credits: Bram van Berlo (b.r.d.v.berlo@tue.nl)
    :param csi_matrix: csi input complex matrix
    :return: dfs of sample
    """
    max_length_ftp = 200  # originally 2000

    # transpose matrix to correct sample
    csi_matrix = np.squeeze(csi_matrix)
    csi_matrix = np.transpose(csi_matrix, [2, 0, 1])

    # Down sample (decimate via factor M, i.e., keep every Mth sample along time dimension)
    # M = 1
    # indices = np.arange(0, csi_matrix.shape[1], M)
    # csi_matrix = csi_matrix[:, indices, :]

    # First, second antenna pair selection (WiDance https://dl.acm.org/doi/pdf/10.1145/3025453.3025678)
    amplitude_mean = np.mean(np.absolute(csi_matrix), axis=1)
    amplitude_var = np.sqrt(np.var(np.absolute(csi_matrix), axis=1))
    mean_var_ratio = np.divide(amplitude_mean, amplitude_var)
    mean_var_ratio = np.mean(mean_var_ratio, axis=1)
    max_idx = np.argmax(mean_var_ratio)
    csi_matrix_ref = np.stack([csi_matrix[max_idx]] * 3)
    max_idx += 1

    # Antenna power adjustment (IndoTrack https://dl.acm.org/doi/pdf/10.1145/3130940)
    amplitude = np.absolute(csi_matrix)
    amplitude_mask = np.ma.masked_equal(amplitude, value=0.0, copy=False)
    alpha = amplitude_mask.min(axis=1)
    amplitude = amplitude - np.transpose(np.stack([alpha] * amplitude.shape[1]), [1, 0, 2])
    amplitude = np.absolute(amplitude)
    angle = np.angle(csi_matrix)
    csi_matrix = np.multiply(amplitude, np.exp(np.multiply(1j, angle)))

    beta = np.divide(np.multiply(1000, np.sum(alpha)), alpha.size)
    amplitude_2 = np.absolute(csi_matrix_ref)
    amplitude_2 = np.add(amplitude_2, beta)
    angle_2 = np.angle(csi_matrix_ref)
    csi_matrix_ref = np.multiply(amplitude_2, np.exp(np.multiply(1j, angle_2)))

    # Conjugate multiplication (DataPort DFSExtraction matlab script)
    conj_multiplication = np.multiply(csi_matrix, np.conjugate(csi_matrix_ref))
    conj_multiplication = np.transpose(conj_multiplication, [1, 0, 2])
    conj_multiplication = np.reshape(conj_multiplication,
                                     newshape=(conj_multiplication.shape[0],
                                               conj_multiplication.shape[1] * conj_multiplication.shape[2])
                                     )
    conj_multiplication = np.concatenate(
        (conj_multiplication[:, 0:30 * (max_idx - 1)], conj_multiplication[:, 30 * max_idx:90]), axis=-1)

    # Static/high frequency component filtering (DataPort DFSExtraction matlab script)
    [lb, la] = signal.butter(6, 60 / 500, 'low')
    [hb, ha] = signal.butter(3, 2 / 500, 'high')
    conj_multiplication = signal.lfilter(lb, la, conj_multiplication, axis=0)
    conj_multiplication = signal.lfilter(hb, ha, conj_multiplication, axis=0)

    # PCA analysis https://stackoverflow.com/questions/60508233/python-implement-a-pca-using-svd
    # Matlab pca(X) returns principal component coefficients, also known as 'loadings' https://nl.mathworks.com/help/stats/pca.html
    conj_multiplication_1 = conj_multiplication.copy()
    conj_multiplication_1 -= np.mean(conj_multiplication_1, axis=0)
    U, S, Vt = np.linalg.svd(conj_multiplication_1, full_matrices=True)
    V = np.transpose(Vt)

    # Enforce sign convention https://nl.mathworks.com/matlabcentral/answers/300899-sign-difference-between-coeff-pca-x-and-v-svd-x
    max_indices = np.argmax(np.absolute(V), axis=0)
    max_value_signs = np.squeeze(np.sign(np.take_along_axis(V, np.expand_dims(max_indices, axis=0), axis=0)))
    max_value_signs = np.stack([max_value_signs] * V.shape[0])
    V = np.multiply(V, max_value_signs)

    conj_multiplication_pca = np.matmul(conj_multiplication, V[:, 0])

    sample_rate = 1000
    extremities_value = 0.005
    window_size = round(sample_rate / 4 + 1)
    if not window_size % 2:
        window_size += 1

    custom_window = np.exp(np.multiply(np.log(extremities_value), np.power(np.linspace(-1., 1., window_size), 2.)))
    conj_fwindow = np.conj(custom_window)

    lh = (custom_window.shape[0] - 1) // 2
    rangemin = min([round(sample_rate / 2.0), lh])
    timestamps = np.arange(0, conj_multiplication_pca.shape[0], 1)
    tfr = np.zeros((sample_rate, timestamps.shape[0]), dtype=np.complex_)

    starts = -np.min(np.c_[rangemin * np.ones(timestamps.shape),
                           np.arange(timestamps.shape[0]) - 1],
                     axis=1).astype(np.int_)
    ends = np.min(np.c_[rangemin * np.ones(timestamps.shape),
                        conj_multiplication_pca.shape[0] - np.arange(timestamps.shape[0])],
                  axis=1).astype(np.int_)

    for icol in range(tfr.shape[1]):
        start = starts[icol]
        end = ends[icol]
        tau = np.arange(start, end + 1).astype(np.int_)
        index = np.remainder(sample_rate + tau, sample_rate)
        tfr[index, icol] = conj_multiplication_pca[(icol + tau - 1).astype(np.int_)] * \
                           conj_fwindow[(lh + tau).astype(np.int_)]

    freq_time_profile_allfreq = np.fft.fft(tfr, axis=0)
    freq_time_profile_allfreq = np.power(np.absolute(freq_time_profile_allfreq), 2)

    freq_bins_unwrap = np.divide(
        np.append(
            np.arange(0, sample_rate // 2, 1),
            np.arange(-sample_rate // 2, 0, 1)
        ),
        sample_rate
    )
    freq_lpf_selection_array = np.logical_and(
        freq_bins_unwrap <= 60 / sample_rate,
        freq_bins_unwrap >= -60 / sample_rate
    )

    freq_time_profile = freq_time_profile_allfreq[freq_lpf_selection_array, :]
    freq_time_profile = np.fft.fftshift(x=freq_time_profile, axes=(0,))
    freq_time_profile = np.divide(freq_time_profile, np.sum(freq_time_profile, axis=0))
    # Changed the max_length_ftp which was originally 2000
    freq_time_profile = freq_time_profile[:, :max_length_ftp]
    padded_freq_time_profile = np.zeros(shape=(121, max_length_ftp), dtype=np.float64)
    padded_freq_time_profile[:freq_time_profile.shape[0], :freq_time_profile.shape[1]] = freq_time_profile

    return padded_freq_time_profile


def convert_csi_to_dfs(csi: np.ndarray) -> np.ndarray:
    """
    Convert csi data to dfs
    :param csi: numpy array of csi data
    :return: numpy array of dfs data
    """
    print('converting signfi to gaf')
    # make empty matrix to fill output
    length_sample = csi.shape[0]
    csi_dfs = np.zeros(shape=(csi.shape[-1], 121, length_sample))

    start = time.time()
    for i in range(csi.shape[-1]):
        # process 1 sample, csi is shape  (time frame, subcarrriers, antennas, samples)
        processed_sample = process_rx_sample(csi[:, :, :, i])
        csi_dfs[i, :, :] = processed_sample

    print('Processing csi to dfs took: ', time.time() - start)

    return csi_dfs


def convert_signfi_to_dfs(data_file: str, objects: List[str], label_obj: str):
    # read dataset
    data_lab_276_ul = sio.loadmat(os.path.join(input_dir_signfi, data_file))

    # open file
    output_file = os.path.join(root_dir_signfi, 'dfs', f'{data_file}-dfs.hdf5')
    f = h5py.File(output_file, 'w')
    labels = data_lab_276_ul[label_obj]
    # add label directory
    f.create_dataset('label', data=labels, dtype='int8')

    for obj in objects:
        print(f'Running object {obj}')
        csi = data_lab_276_ul[obj]
        # preprocess dataset
        dfs_csi = convert_csi_to_dfs(csi)
        # write data to file
        f.create_dataset(f"{obj}", data=dfs_csi, dtype="float32")

    print('done processing')
    f.close()
