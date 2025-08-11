import os

import pandas as pd


def load_file(root: str, path_file: str) -> pd.DataFrame:
    df = pd.read_csv(os.path.join(root, path_file))
    # extract parameters from root and name file
    a, experiment = os.path.split(root)
    dataset, datatype = os.path.split(a)[1].split('_')
    backbone, _, _, env, _,env_nr, date, _ = path_file.split('_')

    print(dataset, datatype, experiment, backbone, env, env_nr, date)
    df['dataset'] = dataset
    df['datatype'] = datatype
    df['experiment'] = experiment
    df['backbone'] = backbone
    df['env'] = env
    df['split_nr'] = env_nr
    df['time'] = date

    if 'splits_leftout' not in df:
        df['splits_leftout'] = 1

    return df


def gather_evaluation_csv(path_results: str):
    """
    Go over the given results directory and combine all csv files to pandas dataframe
    :param path_results: path to results dir
    :return:
    """
    all_dfs = []
    for root, dirs, files in os.walk(path_results):
        eval_files = [file for file in files if 'evaluation' in file]

        for file in eval_files:
            all_dfs.append(load_file(root, file))

    df_results = pd.concat(all_dfs)

    print(df_results.head())

    df_results.to_csv('evaluation_results.csv', index=False)

    return df_results


if __name__ == '__main__':
    # gather_evaluation_csv(r'E:\OneDrive - TU Eindhoven\Documents\UNI\BEP-JournalResearch\results')
    gather_evaluation_csv(r'../results')
