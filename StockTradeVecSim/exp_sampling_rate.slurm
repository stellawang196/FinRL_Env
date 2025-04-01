#!/bin/bash
#SBATCH --job-name=YonV_RL          # create a short name for your job
#SBATCH --output=slurm-YonV-%A.out  # stdout file
#SBATCH --time=4:05:00            # total run time limit (HH:MM:SS)

#SBATCH --nodes=1              # node count
#SBATCH --ntasks=1             # total number of tasks across all nodes
#SBATCH --cpus-per-task=64     # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=8G       # memory per cpu-core (4G is default)
#SBATCH --gres=gpu:1           # number of GPUs per node

# conda activate go_explore
echo $CONDA_DEFAULT_ENV

START_DATE="$(date)"
echo "|start time: $START_DATE"

#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/yanglet/.mujoco/mujoco210/bin
python3 exp_sampling_rate.py "/colab_space/yanglet/results_$SLURM_JOB_ID"

END_DATE="$(date)"
echo "|start time: $START_DATE"
echo "|end time:   $END_DATE"
