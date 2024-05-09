#!/bin/bash
#SBATCH --nodes=1               # node count
#SBATCH -n 1                    # number of cores
#SBATCH -p gpu --gres=gpu:1     # number of gpus per node
#SBATCH --mem-per-cpu=32G       # total memory per node (4 GB per cpu-core is default)
#SBATCH -t 48:00:00             # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin       # send email when job begins
#SBATCH --mail-type=end         # send email when job ends
#SBATCH --mail-user=noah_rousell@brown.edu

module purge
unset LD_LIBRARY_PATH

cd src

export APPTAINER_BINDPATH="/oscar/home/$USER,/oscar/scratch/$USER,/oscar/data"
export PYTHONUNBUFFERED=TRUE


srun apptainer exec --nv ../tensorflow-24.03-tf2-py3.simg python -m main --train --seed 3 --name $SLURM_JOB_NAME