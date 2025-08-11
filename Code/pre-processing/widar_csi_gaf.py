import itertools
import multiprocessing
import os

import cv2
import h5py
import numpy as np
from CSIKit.reader import IWLBeamformReader
from pyts.image import GramianAngularField
from scipy import signal

input_root_widar = 'not-processed/'
output_root_widar = 'processed/'


def process_files_user(current_root, files_one_item):
    first_item = files_one_item[0]

    # process input file name to get user, room, orientation, etc..
    item_name_split = first_item.split("-")
    item_name_split[0] = item_name_split[0].replace("user", "")
    item_name_split[-1] = item_name_split[-1].replace(".dat", "")
    item_name_split[-1] = item_name_split[-1].replace("r", "")
    item_name_split = list(map(int, item_name_split))

    current_user = current_root.replace(input_root_widar, '').split(os.sep)[1]

    if current_user == "20181109":
        room_label = 1

        item_name_split[1] = 10 if item_name_split[1] == 5 else (
            11 if item_name_split[1] == 6 else item_name_split[1]
        )
    elif current_user == "20181112" or current_user == "20181116":
        room_label = 1

        item_name_split[1] += 12
    elif current_user == "20181115":
        room_label = 1

        item_name_split[1] = 12 if item_name_split[1] == 4 else (
            10 if item_name_split[1] == 5 else (
                11 if item_name_split[1] == 6 else item_name_split[1]
            )
        )
    elif current_user == "20181117" or current_user == "20181118":
        room_label = 2

        item_name_split[1] = 10 if item_name_split[1] == 5 else (
            11 if item_name_split[1] == 6 else (
                12 if item_name_split[1] == 4 else item_name_split[1]
            )
        )
    elif current_user == "20181121" or current_user == "20181127":
        room_label = 1
        if current_user == "20181127":
            room_label = 2

        item_name_split[1] = 4 if item_name_split[1] == 1 else (
            6 if item_name_split[1] == 2 else (
                9 if item_name_split[1] == 3 else (
                    5 if item_name_split[1] == 4 else (
                        8 if item_name_split[1] == 5 else (
                            7 if item_name_split[1] == 6 else item_name_split[1]
                        )
                    )
                )
            )
        )
    elif current_user == "20181128":
        room_label = 2

        item_name_split[1] = 6 if item_name_split[1] == 4 else (
            9 if item_name_split[1] == 5 else (
                5 if item_name_split[1] == 6 else item_name_split[1]
            )
        )
    elif current_user == "20181130" or current_user == "20181204":
        room_label = 1
        if current_user == "20181204":
            room_label = 2

        item_name_split[1] = 6 if item_name_split[1] == 5 else (
            9 if item_name_split[1] == 6 else (
                5 if item_name_split[1] == 7 else (
                    7 if item_name_split[1] == 9 else item_name_split[1]
                )
            )
        )
    elif current_user == "20181205":
        room_label = 2
        if item_name_split[0] == 2:
            item_name_split[1] = 6 if item_name_split[1] == 1 else (
                9 if item_name_split[1] == 2 else (
                    5 if item_name_split[1] == 3 else (
                        8 if item_name_split[1] == 4 else (
                            7 if item_name_split[1] == 5 else item_name_split[1]
                        )
                    )
                )
            )
        elif item_name_split[0] == 3:
            item_name_split[1] = 4 if item_name_split[1] == 1 else (
                6 if item_name_split[1] == 2 else (
                    9 if item_name_split[1] == 3 else (
                        5 if item_name_split[1] == 4 else (
                            8 if item_name_split[1] == 5 else (
                                7 if item_name_split[1] == 6 else item_name_split[1]
                            )
                        )
                    )
                )
            )
    elif current_user == "20181208":
        room_label = 2
    elif current_user == "20181209":
        room_label = 2

        if item_name_split[0] == 6:
            item_name_split[1] = 6 if item_name_split[1] == 5 else (
                9 if item_name_split[1] == 6 else item_name_split[1]
            )
    elif current_user == "20181211":
        room_label = 3

        item_name_split[1] = 6 if item_name_split[1] == 5 else (
            9 if item_name_split[1] == 6 else item_name_split[1]
        )
    else:
        raise Exception(f"An unknown data collection date subfolder was encountered, {current_user}")

    # Skip specific domains/gestures for correct domain factor leave out cross-validation due to distribution imbalances
    # --------------------4500 samples (6 users 5 positions 5 orientations 6 gestures 5 instances--------------------- #

    # Limit number of repetitions per domain in the dataset (reduces size for hyperparameter tuning)
    # if item_name_split[4] > 1:
    #    return

    if any(item_name_split[0] == x for x in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]):
        return
    if any(item_name_split[1] == x for x in [5, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]):
        return
    if any(item_name_split[2] == x for x in [6, 7, 8]):
        return

    item_name_split[0] = item_name_split[0] - 11

    item_name_split[1] = 5 if item_name_split[1] == 6 else (
        6 if item_name_split[1] == 9 else item_name_split[1]
    )

    # One-hot vector of which every index denotes unique (user, room, position, orientation) pair
    # Room label has been defined in if elif else ladder
    domain_label = np.zeros((6, 1, 5, 5), dtype=np.int8)
    domain_label[
        item_name_split[0] - 1,
        room_label - 1,
        item_name_split[2] - 1,
        item_name_split[3] - 1
    ] = 1
    domain_label = domain_label.flatten()

    task_label = np.zeros(6, dtype=np.int8)
    task_label[item_name_split[1] - 1] = 1

    try:
        gaf_stacked = np.stack(
            arrays=[process_widar_sample_to_gaf(current_root, x) for x in files_one_item],
            axis=2)
    except Exception as e:
        print(f'Exception on files {files_one_item}')
        raise e
    gaf_resized = cv2.resize(gaf_stacked, (500, 500))

    lock.acquire()
    # print(f'new sample {gaf_resized.shape}')

    with h5py.File(f'{output_root_widar}/widar3.0-domain-leave-out-dataset-gaf.hdf5', 'a') as f:
        if 'task_labels' in f and 'domain_labels' in f and 'inputs' in f:
            dset_1 = f['inputs']
            dset_2 = f['task_labels']
            dset_3 = f['domain_labels']

            dset_1.resize(dset_1.shape[0] + 1, axis=0)
            dset_2.resize(dset_2.shape[0] + 1, axis=0)
            dset_3.resize(dset_3.shape[0] + 1, axis=0)

            dset_1[-1] = gaf_resized
            dset_2[-1] = task_label
            dset_3[-1] = domain_label
        else:
            dset_1 = f.create_dataset("inputs", (1, *gaf_resized.shape), dtype="float32",
                                      maxshape=(None, *gaf_resized.shape))
            dset_2 = f.create_dataset("task_labels", (1, 6), dtype="int8", maxshape=(None, 6))
            dset_3 = f.create_dataset("domain_labels", (1, 150), dtype="int8", maxshape=(None, 1275))

            dset_1[0] = gaf_resized
            dset_2[0] = task_label
            dset_3[0] = domain_label

    lock.release()


def process_widar_sample_to_gaf(a_root, item):
    """
    Convert a single widar csi sample to gaf representation, includes other preprocessing
    :param a_root: current file root
    :param item: item in folder to process
    :return: numpy array for
    """

    # Reading raw CSI data
    reader_obj = IWLBeamformReader()
    data_obj = reader_obj.read_file(a_root + os.sep + item)
    csi_matrix = np.stack([x.csi_matrix for x in data_obj.frames])

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

    # convert to batch size for gaf
    conj_multiplication_pca = conj_multiplication_pca.reshape(1, -1)

    # convert to amplitude
    conj_multiplication_pca = np.abs(conj_multiplication_pca)

    # compute GAF
    gadf = GramianAngularField(method='difference')
    csi_gaf = gadf.fit_transform(conj_multiplication_pca)

    csi_gaf = np.squeeze(csi_gaf)
    # truncate and pad images to be 2000, 2000
    csi_gaf = csi_gaf[:2000, :2000]

    csi_gaf = np.pad(csi_gaf, ((0, 2000 - csi_gaf.shape[0]), (0, 2000 - csi_gaf.shape[1])))

    return csi_gaf


def filter_list_item(list_item_root, list_item):
    if '.baiduyun.uploading.cfg' in list_item or '.dat' not in list_item:
        return False

    user_part = list_item_root.replace(input_root_widar, '').split(os.sep)[1]

    if user_part == "20181109" and any(x in list_item for x in ['user2-6-4-4-2-', 'user3-1-3-1-8-']):
        return False
    elif user_part == "20181118" and 'user2-3-5-3-4-' in list_item:
        return False
    elif user_part == "20181209" and 'user6-3-1-1-5-' in list_item:
        return False
    elif user_part == "20181211" and any(
            x in list_item for x in ['user8-1-1-1-1-', 'user8-3-3-3-5-', 'user9-1-1-1-1-']):
        return False

    return True


def init(a_lock):
    global lock
    lock = a_lock


def create_widar_gaf_dataset():
    main_lock = multiprocessing.Lock()

    # with multiprocessing.Pool(processes=multiprocessing.cpu_count(), initializer=init, initargs=(main_lock,)) as p:
    with multiprocessing.Pool(processes=4, initializer=init, initargs=(main_lock,)) as p:
        # Local test directory: Widar3.0-CSI
        for root, _, files in os.walk(input_root_widar, topdown=True):
            if len(files) == 0:
                continue
            # only do original data subfolders
            folder = root.replace(input_root_widar, '')
            if folder == '' or not folder.split(os.sep)[1].startswith('2018'):
                continue
            print(f'Processing root: {root}')

            files.sort()
            files = list(filter(lambda x: filter_list_item(root, x), files))

            if len(files) != 0:
                files = iter(files)
                grouped_files = iter(lambda: list(itertools.islice(files, 6)), [])
                p.starmap(func=process_files_user, iterable=zip(itertools.repeat(root), grouped_files))
                print(f'Done with root: {root}')

    print('Done processing, widar')


if __name__ == '__main__':
    create_widar_gaf_dataset()
