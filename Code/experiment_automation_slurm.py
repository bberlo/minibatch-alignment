import argparse
import datetime
import pickle
import uuid
import os

from sklearn.model_selection import KFold
from simple_slurm import Slurm
import numpy as np

from custom_items.data_specs import domain_order_dataset, domain_size_datasets, SYSTEM
from custom_items.utilities import domain_to_be_left_out_indices_calculation
from custom_items.data_fetching import fetch_labels_indices, get_dir_data

# Command prompt settings for experiment automation
parser = argparse.ArgumentParser(description="Experiment automation setup script.")
parser.add_argument("-m_n", "--model_name", help="<Required> Set model name to be used in the experiment", required=True)
parser.add_argument("-t", "--type", help="<Required> Experiment type: tuning, testing or testing-leave-out", required=True,
                    choices=["tuning", "tuning-testing", "testing-leave-out"])
parser.add_argument("-g", "--gpu", type=int, help="GPU to be used per experiment", required=False)
parser.add_argument("-d", "--dataset", help="<Required> Which dataset to run experiments on", required=True,
                    choices=["signfi", "widar"])
parser.add_argument("-d_t", "--datatype", help="<Required> Which datatype of the dataset to be used either dfs or gaf", required=True,
                    choices=["dfs", "gaf"])
parser.add_argument("-bb", "--backbone", default="efficient", help="<Required> Which backbone to use", required=True,
                    choices=["efficientnet", "resnet", "vgg"])
parser.add_argument("-do_t", "--domain_type", help="Domain type: user, position, orientation, environment", required=False,
                    choices=["user", "position", "orientation", "environment"])
parser.add_argument("-lft", "--splits_leftout", default=1, help="Number of domains to leave out for leave-out experiments")
parser.add_argument("-cvs", "--crossval_split", help="if you want to run a specific split", required=False)
args = parser.parse_args()

# Other configurations part 1
path_dataset = get_dir_data(args.dataset, args.datatype, args.domain_type, local=False)
dataset_domain_size = domain_size_datasets[args.dataset]
domain_types = list(range(1, dataset_domain_size + 1, 1))

# Fetch domain labels
domain_labels = fetch_labels_indices(f_path=path_dataset)
domain_labels = np.argmax(domain_labels, axis=1) + 1

# Check if relative tmp/ directory is available
os.makedirs("tmp", exist_ok=True)


def get_test_types_leave_out():
    """
    Get the domain types list for the given domain factor leave out experiment
    :return:
    """

    domain_label_struct = domain_order_dataset[args.dataset]["struct"]
    CUR_DOMAIN_FACTOR = domain_order_dataset[args.dataset][args.domain_type]
    CUR_DOMAIN_FACTOR_TOTAL_NR = domain_label_struct[domain_order_dataset[args.dataset][args.domain_type]]

    if args.domain_type == "orientation" and args.splits_leftout == 2:
        CUR_DOMAIN_FACTOR_TOTAL_NR = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 4]]
        test_types_list = [
            list(
                map(
                    lambda y: y + 1,
                    domain_to_be_left_out_indices_calculation(CUR_DOMAIN_FACTOR, x[0], domain_label_struct),
                )
            )
            + list(
                map(
                    lambda y: y + 1,
                    domain_to_be_left_out_indices_calculation(CUR_DOMAIN_FACTOR, x[1], domain_label_struct),
                )
            )
            for x in CUR_DOMAIN_FACTOR_TOTAL_NR
        ]
    else:
        # First func. variable needs to match loc. of CUR_TOTAL_DOMAIN_FACTOR_NR in quadruple
        test_types_list = [
            list(
                map(
                    lambda y: y + 1,
                    domain_to_be_left_out_indices_calculation(CUR_DOMAIN_FACTOR, x, domain_label_struct),
                )
            )
            for x in range(CUR_DOMAIN_FACTOR_TOTAL_NR)
        ]

    return test_types_list


if args.type == "tuning":
    CUR_DOMAIN_FACTOR_NAME = "random"
    f_name = "tuning_supervised.py"

    kfold_obj = KFold(n_splits=6, shuffle=True, random_state=42)
    test_types_indices_list = [x[1].tolist() for x in kfold_obj.split(domain_types)]
    test_types_list = [[domain_types[x] for x in z] for z in test_types_indices_list]
    test_types_list = test_types_list[:1]  # limit tuning process to one split

elif args.type == "tuning-testing":
    CUR_DOMAIN_FACTOR_NAME = "random"
    f_name = "experiment_test.py"

    kfold_obj = KFold(n_splits=6, shuffle=True, random_state=42)
    test_types_indices_list = [x[1].tolist() for x in kfold_obj.split(domain_types)]
    test_types_list = [[domain_types[x] for x in z] for z in test_types_indices_list]

elif args.type == "testing-leave-out":
    CUR_DOMAIN_FACTOR_NAME = args.domain_type
    f_name = "cross_validation_experiment.py"

    test_types_list = get_test_types_leave_out()

else:
    raise ValueError("Unknown experiment type was given as argument.")

# Other configurations part 2 (last part)
split_nrs = range(len(test_types_list))
GPU_DEVICE = args.gpu
learning_rate = 0.0001

# Slurm cluster configuration
if SYSTEM == "snellius":
    cluster_config_obj = Slurm(
        job_name=args.model_name,
        nodes=1,
        ntasks=1,
        cpus_per_task=18,
        gpus=1,
        partition="gpu",
        time=datetime.timedelta(hours=120),
        error="slurm-%j.err",
        output="slurm-%j.out",
    )
    load_modules = "\n".join(
        [
            "module load 2022",
            "module load Python/3.10.4-GCCcore-11.3.0",
            "module load CUDA/11.7.0",
            "module load cuDNN/8.4.1.50-CUDA-11.7.0"
        ]
    )

else:  # SYSTEM settings for TU/e WIN HPC
    cluster_config_obj = Slurm(
        "--job_name", args.model_name,
        "--nodes", "1",
        "--ntasks", "3",
        '--partition', 'mcs.gpu.q',
        "--error", "slurm-%j.err",
        "--output", "slurm-%j.out",
        "--time", "10-0",  # DD-HH comb., HH:MM:SS comb.: '--time', "01:00:00"
        "--constraint", "2080",  # other node is 2080ti
        "--gres", "gpu:1"
    )
    load_modules = ""

for split_nr, test_types in zip(split_nrs, test_types_list):

    if args.crossval_split is not None:  # Skip splits if specific experiment split has been set
        if int(args.crossval_split) != split_nr:
            continue

    train_types = list(set(range(1, dataset_domain_size + 1, 1)) - set(test_types))
    train_indices, test_indices = (
        np.where(np.isin(domain_labels, test_elements=np.asarray(train_types)))[0],
        np.where(np.isin(domain_labels, test_elements=np.asarray(test_types)))[0],
    )

    # pickle the indices and types for the sub process to run experiment on
    file_path = "tmp/" + uuid.uuid4().hex + ".pickle"
    with open(file_path, "wb") as handle:
        pickle.dump(
            obj={"train_indices": train_indices, "test_indices": test_indices, "train_types": train_types},
            file=handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    python_command = " ".join(
        [
            "python", f_name,
            "-e_s", "100",
            "-b_s", "12",
            "-d_f_n", CUR_DOMAIN_FACTOR_NAME,
            "-cv_s", str(split_nr),
            "-g", str(GPU_DEVICE),
            "-l_r", str(learning_rate),
            "-m_n", args.model_name,
            "-f_p", file_path,
            "-d", args.dataset,
            "-d_t", args.datatype,
            "-bb", args.backbone,
            "-lft", args.splits_leftout,
        ]
    )
    cluster_config_obj.sbatch(run_cmd=load_modules + python_command, shell="/bin/bash")
