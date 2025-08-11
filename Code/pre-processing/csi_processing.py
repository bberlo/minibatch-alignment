from subprocess import run
import argparse

parser = argparse.ArgumentParser(description='Convert CSI dataset to gaf, slurm script.')
parser.add_argument('-d', '--dataset', help='<Required> Which dataset to convert, widar or signfi',
                    required=True, choices=['widar', 'signfi'])
parser.add_argument('-dt', '--datatype', help='<Required> Which datatype to convert to',
                    required=True, choices=['dfs', 'gaf'])
args = parser.parse_args()


def main():
    if args.dataset == 'widar':

        if args.datatype == 'dfs':
            # How to process widar dfs? See Berlo-Bram van -> 2022 -> PerFail2022 -> pre-processing directory
            print('not implemented')

        elif args.datatype == 'gaf':

            with open('{}_{}_preprocess_stdout.txt'.format(args.dataset, args.datatype), 'w') as stdoutFile, \
                 open('{}_{}_preprocess_stderr.txt'.format(args.dataset, args.datatype), 'w') as stderrFile:

                run(['python', 'widar_csi_gaf.py'], stdout=stdoutFile, stderr=stderrFile)

    elif args.dataset == 'signfi':

        with open('{}_{}_preprocess_stdout.txt'.format(args.dataset, args.datatype), 'w') as stdoutFile, \
             open('{}_{}_preprocess_stderr.txt'.format(args.dataset, args.datatype), 'w') as stderrFile:

            run(['python', 'signfi_processing.py',
                 '-dt', args.datatype], stdout=stdoutFile, stderr=stderrFile)


if __name__ == "__main__":
    main()
