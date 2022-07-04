####### This script is not complete and should be revised. ###########

#!/bin/bash
####### Reserve computing resources #############
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=10:00:00
#SBATCH --mem=1G
#SBATCH --partition=cpu2019

####### Run your script #########################
pip install -r requirements.txt
python main_script.py +configs=settings_1
python main_script.py +configs=settings_2
python main_script.py +configs=settings_3