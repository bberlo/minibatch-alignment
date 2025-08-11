import argparse
import os

from simple_slurm import Slurm

# Command prompt settings for experiment automation
parser = argparse.ArgumentParser(description='Small automation scripts that need to be run on cluster.')
parser.add_argument('-a', '--action', help='<Required> Which action to perform', required=True,
                    choices=['check_disk_space', 'move_gaf_to_local', 'move_dfs_to_local'])

args = parser.parse_args()

cluster_config_obj = Slurm(
    '--job_name', 'test_local_size',
    '--nodes', '1',
    '--ntasks', '1',
    '--partition', 'mcs.irisgpu.q',
    '--error', 'slurm-%j.err',
    '--output', 'slurm-%j.out',
    '--time', '01:00:00',
    '--constraint', '2080',  # other nodes is 2080ti
    '--gres', 'gpu:1'
)

if args.action == 'check_disk_space':
    cluster_config_obj.sbatch(run_cmd='python check_disk_space.py', shell='/bin/bash')
elif args.action == 'move_gaf_to_local':
    path_gaf_set = '/home/mcs001/20184025/data/Widar/widar3.0-domain-leave-out-dataset-gaf.hdf5'
    path_to_move_to = '/local/20184025/'
    os.makedirs(path_to_move_to, exist_ok=True)
    cluster_config_obj.sbatch(run_cmd=f'rsync -vh --progress {path_gaf_set} {path_to_move_to}', shell='/bin/bash')
elif args.action == 'move_dfs_to_local':
    path_gaf_set = '/home/mcs001/20184025/data/Widar/widar3.0-domain-leave-out-dataset-benchmark-2.hdf5'
    path_to_move_to = '/local/20184025/'
    os.makedirs(path_to_move_to, exist_ok=True)
    cluster_config_obj.sbatch(run_cmd=f'rsync -vh --progress {path_gaf_set} {path_to_move_to}', shell='/bin/bash')
