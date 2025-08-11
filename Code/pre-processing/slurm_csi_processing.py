import argparse

from simple_slurm import Slurm

parser = argparse.ArgumentParser(description='Convert CSI dataset to gaf, slurm script.')
parser.add_argument('-d', '--dataset', help='<Required> Which dataset to convert, widar or signfi',
                    required=True, choices=['widar', 'signfi'])
parser.add_argument('-dt', '--datatype', help='<Required> Which datatype to convert to',
                    required=True, choices=['dfs', 'gaf'])
args = parser.parse_args()

# Slurm cluster configuration
cluster_config_obj = Slurm(
    '--job_name', 'widar_csi_to_gaf',
    '--nodes', '1',
    '--ntasks', '47' if args.dataset == 'widar' else '1',
    '--partition', 'mcs.default.q',
    '--error', 'slurm-%j.err',
    '--output', 'slurm-%j.out',
    '--time', '24:00:00' if args.dataset == 'widar' else '04:00:00',
)


def main():
    if args.dataset == 'widar':
        if args.datatype == 'dfs':
            # How to process widar dfs? See Berlo-Bram van -> 2022 -> PerFail2022 -> pre-processing directory
            print('not implemented')
        elif args.datatype == 'gaf':
            cluster_config_obj.sbatch(run_cmd=' '.join(['python', 'widar_csi_gaf.py']), shell='/bin/bash')

    elif args.dataset == 'signfi':
        cluster_config_obj.sbatch(run_cmd=' '.join(['python', 'signfi_processing.py', '-dt', args.datatype]),
                                  shell='/bin/bash')


if __name__ == "__main__":
    main()
