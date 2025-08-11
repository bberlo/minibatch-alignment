import argparse


def model_args_parser() -> argparse.ArgumentParser:
    """
    Create the default model argument parser
    :return:
    """
    parser = argparse.ArgumentParser(description="Experiment automation setup script.")
    parser.add_argument('-e_s', '--epoch_size', type=int, help='<Required> Set epoch size to be used in the experiment',
                        required=True)
    parser.add_argument('-b_s', '--batch_size', type=int, help='<Required> Set batch size to be used in the experiment',
                        required=True)
    parser.add_argument('-g', '--gpu', type=int, help='<Required> Set GPU device to be used in the experiment',
                        required=True)
    parser.add_argument('-cv_s', '--crossval_split', type=int,
                        help='<Required> Current cross val. split used in the experiment', required=True)
    parser.add_argument('-l_r', '--learning_rate', type=float,
                        help='<Required> Learning rate to be used in the experiment', required=True)
    parser.add_argument('-m_n', '--model_name', help='<Required> Set model name to be used in the experiment',
                        required=True)
    parser.add_argument('-d_f_n', '--domain_factor_name',
                        choices=['user', 'position', 'orientation', 'environment', 'random'],
                        help='<Required> Set domain factor name to be used in the experiment', required=True)
    parser.add_argument('-f_p', '--file_path', help='<Required> File path to be used in the experiment', required=True)
    parser.add_argument('-d', '--dataset', required=True, choices=['signfi', 'widar'],
                        help='<Required> Which dataset to run experiments on')
    parser.add_argument('-d_t', '--datatype', required=True, choices=['dfs', 'gaf'],
                        help='<Required> Which datatype of the dataset to be used either dfs or gaf')
    parser.add_argument('-bb', '--backbone', required=True, choices=['efficientnet', 'resnet', 'vgg'],
                        default='efficient', help='<Required> Which backbone to use')
    parser.add_argument("-lft", "--splits_leftout", default=1,
                        help="Number of domains to leave out for leave-out experiments")

    return parser
