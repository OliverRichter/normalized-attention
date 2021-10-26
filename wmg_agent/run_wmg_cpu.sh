#!/bin/bash

#SBATCH --mail-type=END                     # mail configuration: NONE, BEGIN, END, FAIL, REQUEUE, ALL
#SBATCH --output=/home/disco-computing/richtero/log/%j.out                 # where to store the output (%j is the JOBID), subdirectory must exist
#SBATCH --error=/home/disco-computing/richtero/log/%j.err                 # where to store error messages
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-399

/bin/echo Running on host: `hostname`
/bin/echo In directory: `pwd`
/bin/echo Starting on: `date`
/bin/echo SLURM_JOB_ID: $SLURM_JOB_ID
# source activate wmg_env

# exit on errors
set -o errexit

# binary to execute
# e.g. python main.py --input large.yaml
python run.py $1 $SLURM_ARRAY_TASK_ID

echo finished at: `date`
exit 0;
