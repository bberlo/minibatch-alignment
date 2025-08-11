# MDPI Sensors journal paper - Mini-Batch Alignment: A Deep-Learning Model for Domain Factor-Independent Feature Extraction for Wi-Fi–CSI Data
Author: Bram van Berlo - b.r.d.v.berlo@tue.nl

### Requirements
In the requirements.txt file the needed python modules for this project are specified.

## Preprocessing steps
The pre-processed datasets in '../Data' are created using the pre-processing scripts in 'pre-processing/'.
Links to the original datasets have been added to this subdirectory as well.
The pre-processing scripts are not part of the audit procedure and can therefore not be run automatically.

Steps required in order to use the pre-processing scripts:

1) Download entire original dataset to a directory.
2) Download '../Code' subdirectory to a directory.
3) Using python 3.7 < ver. < 3.9, install imported packages listed in the pre-processing scripts.

Depending on script arguments with which csi_processing.py in pre-processing/ is run with, follow either one of the following set of steps.

### -d widar -dt dfs

This combination is not supported. Please consider the pre-processing procedure inside the Bram-Berlo van/ -> 2022/ -> PerFail2022/ -> Code/ -> pre-processing/ directory. More specifically, follow the pre-processing steps in the Code/README.txt together with subsection 'Widar_DFS_create_npy_files_domain_leave_out.py'.

### -d widar -dt gaf

4) Extract all CSI_*.zip archives into the pre-processing/not-processed/ subdirectory.
5) list_item_root.split(os.sep)[1] and current_root.split(os.sep)[1] calls depend on specific number of directories inside the path string. These calls should be checked on if list index 1 returns date substrings. If not, list index should be updated.
6) Execute csi_processing.py pre-processing script with aforementioned script arguments.
7) Pre-processed datasets can be found inside the pre-processing/processed/ subdirectory.

### -d signfi -dt dfs

4) Put dataset_lab_276_dl.mat, dataset_home_276.mat, and dataset_lab_150.mat in pre-processing/not-processed/ subdirectory.
5) Execute csi_processing.py pre-processing script with aforementioned script arguments.
6) Pre-processed datasets can be found inside the pre-processing/processed/ subdirectory.

### -d signfi -dt gaf

4) Put dataset_lab_276_dl.mat, dataset_home_276.mat, and dataset_lab_150.mat in pre-processing/not-processed/ subdirectory.
5) Execute csi_processing.py pre-processing script with aforementioned script arguments.
6) Pre-processed datasets can be found inside the pre-processing/processed/ subdirectory.

# Reproducing results

The run.sh file includes all bash commands which should be run to acquire the results used in Figures 3-6 of the journal paper.
Prior to running the bash commands, make sure that all packages listed in the requirements.txt file are installed.
Prior to running the bash commands, copy the datasets inside the '../Data' directory to the 'Datasets/' directory.
The results in .csv format are placed in separate *evaluation*.csv (A, P, F, R, CK values measured once on left-out test subset) and *train_history*.csv (train/val subset losses, train/val subset A, P, F, R, CK values per epoch) files. The files are ordered in results/{args.dataset}_{args.datatype}/{args.model_name}/{args.backbone} subdirectories.

## Plots

Figure plots inside the journal paper were created by processing the .csv formatted results with MS Excel into chart structures.

### Figures 3-5

Data in the *evaluation*.csv files should be grouped per dataset (widar3, signfi), data type (dfs, gaf), deep learning domain shift mitigation technique, and if one or two domain factors are being left out in the test subset. 
On the grouped data, AVERAGE and VAR.S functions should be called per metric.
The function outputs should be structured according to chart structures comparable to the structures used inside Figures 3-5 of the journal paper.

### Figure 6

Data in the *train_history*.csv files should be grouped per dataset (widar3, signfi) and data type (dfs, gaf) for the fido domain shift mitigation technique and user 3 (range 0-4) being left out in the test subset.
The overall losses ('loss' and 'val_loss' columns) should be plotted inside a line chart.
The line charts should be organized in the grid which can be found in Figure 6 of the journal paper.
