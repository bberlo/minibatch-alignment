# MDPI Sensors journal paper - Mini-Batch Alignment: A Deep-Learning Model for Domain Factor-Independent Feature Extraction for Wi-Fi–CSI Data
Author: Bram van Berlo - b.r.d.v.berlo@tue.nl

## Use of the docker container
Note: steps listed below only work on a host machine running with a Linux OS.

1) Download the entire 'MDPISensorsJournal-23-23-9534' directory to a machine.

Call the following docker commands inside the 'MDPISensorsJournal-23-23-9534' directory:

2) docker build -t bberlo/mdpi-audit:2023.11 .
3) docker run --name bberlo_mdpi_audit -v $PWD/Data/:/project/Code/Datasets/ bberlo/mdpi-audit:2023.11 /bin/bash /project/Code/run.sh
4) (inside a new terminal) docker cp bberlo_mdpi_audit:/project/Code/results/ ./Code/results

Note: wait for a docker command to be finished running before executing the next docker command.
Note: In order to check if the docker run command is finished, inspect the terminal information of the run.sh command (docker run is not executed in detached mode).

## Plots

Please check the readme file inside the 'Code' directory on how the figure plots (Figures 3-6) inside the journal paper should be created using the data inside the 'Code/results' directory.
