#!/bin/bash
# Template Author(s): James Owers (james.f.owers@gmail.com)
#
# example usage:
# ```
# EXPT_FILE=experiments.txt  # <- this has a command to run on each line
# NR_EXPTS=`cat ${EXPT_FILE} | wc -l`
# MAX_PARALLEL_JOBS=12 
# sbatch --array=1-${NR_EXPTS}%${MAX_PARALLEL_JOBS} slurm_arrayjob.sh $EXPT_FILE
# ```
#
# or, equivalently and as intended, with provided `run_experiement`:
# ```
# run_experiment -b git/memitpp/experiment-scripts/exp_gpt2medium.sh -e git/memitpp/experiment-scripts/exp_gpt2medium.txt -m 40
# run_experiment -b git/memitpp/experiment-scripts/exp_gpt2medium.sh -e git/memitpp/experiment-scripts/exp_gpt2medium_rerun3.txt -m 1
# run_experiment -b git/memitpp/experiment-scripts/exp_gpt2medium.sh -e git/memitpp/experiment-scripts/testexp.txt -m 20
# ```

# ====================
# Options for sbatch
# ====================

# Location for stdout log - see https://slurm.schedmd.com/sbatch.html#lbAH
#SBATCH --output=/home/%u/slurm_logs/slurm-%A_%a.out

# Location for stderr log - see https://slurm.schedmd.com/sbatch.html#lbAH
#SBATCH --error=/home/%u/slurm_logs/slurm-%A_%a.out

# Maximum number of nodes to use for the job
# #SBATCH --nodes=10

# Generic resources to use - typically you'll want gpu:n to get n gpus
##SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=single:1
#SBATCH --gres=gpu:a6000

# Megabytes of RAM required. Check `cluster-status` for node configurations
#SBATCH --mem=20000

# Number of CPUs to use. Check `cluster-status` for node configurations
#SBATCH --cpus-per-task=2

# Maximum time for the job to run, format: days-hours:minutes:seconds
#SBATCH --time=4:00:00

##parameters
export MODEL=models--gpt2-medium
export MODEL_NAME=gpt2-medium

# =====================
# Logging information
# =====================

# slurm info - more at https://slurm.schedmd.com/sbatch.html#lbAJ
echo "Job running on ${SLURM_JOB_NODELIST}"

dt=$(date '+%d/%m/%Y %H:%M:%S')
echo "Job started: $dt"


# ===================
# Environment setup
# ===================

echo "Setting up bash enviroment"

# Make available all commands on $PATH as on headnode
source ~/.bashrc

# Make script bail out after first error
set -e

# Make your own folder on the node's scratch disk
# N.B. disk could be at /disk/scratch_big, or /disk/scratch_fast. Check
# yourself using an interactive session, or check the docs:
#     http://computing.help.inf.ed.ac.uk/cluster-computing
SCRATCH_DISK=/disk/scratch
SCRATCH_HOME=${SCRATCH_DISK}/${USER}
mkdir -p ${SCRATCH_HOME}

# Activate your conda environment
CONDA_ENV_NAME=memit
echo "Activating conda environment: ${CONDA_ENV_NAME}"
conda activate ${CONDA_ENV_NAME}

#setup python path
export PYTHONPATH=/home/${USER}/git/memitpp:${PYTHONPATH}

# =================================
# Move input data to scratch disk
# =================================
# Move data from a source location, probably on the distributed filesystem
# (DFS), to the scratch space on the selected node. Your code should read and
# write data on the scratch space attached directly to the compute node (i.e.
# not distributed), *not* the DFS. Writing/reading from the DFS is extremely
# slow because the data must stay consistent on *all* nodes. This constraint
# results in much network traffic and waiting time for you!
#
# This example assumes you have a folder containing all your input data on the
# DFS, and it copies all that data  file to the scratch space, and unzips it. 
#
# For more guidelines about moving files between the distributed filesystem and
# the scratch space on the nodes, see:
#     http://computing.help.inf.ed.ac.uk/cluster-tips

echo "Moving input data to the compute node's scratch space: $SCRATCH_DISK"

#moving data from DFS to scratch
repo_home=/home/${USER}/git/memitpp

#Moving data
src_path=${repo_home}/data/
dest_path=${SCRATCH_HOME}/memitpp/data
mkdir -p ${dest_path}  # make it if required
echo "Moving data from ${src_path} to ${dest_path}"
rsync --archive --update --compress --progress --verbose --log-file=/dev/stdout ${src_path}/ ${dest_path}

#Moving hparams
src_path=${repo_home}/hparams/
dest_path=${SCRATCH_HOME}/memitpp/hparams
mkdir -p ${dest_path}  # make it if required
echo "Moving data from ${src_path} to ${dest_path}"
rsync --archive --update --compress --progress --verbose --log-file=/dev/stdout ${src_path}/ ${dest_path}

##Moving huggingface hub cache
src_path=/home/${USER}/.cache/huggingface/hub/${MODEL}
dest_path=${SCRATCH_HOME}/memitpp/data/huggingface/hub/${MODEL}
mkdir -p ${dest_path}  # make it if required
echo "Moving data from ${src_path} to ${dest_path}"
rsync --archive --update --compress --progress --verbose --log-file=/dev/stdout ${src_path}/ ${dest_path}

#Set huggingface cache to scratch
export HF_DATASETS_CACHE=${SCRATCH_HOME}/memitpp/data/huggingface/datasets
export HUGGINGFACE_HUB_CACHE=${SCRATCH_HOME}/memitpp/data/huggingface/hub

# Important notes about rsync:
# * the --compress option is going to compress the data before transfer to send
#   as a stream. THIS IS IMPORTANT - transferring many files is very very slow
# * the final slash at the end of ${src_path}/ is important if you want to send
#   its contents, rather than the directory itself. For example, without a
#   final slash here, we would create an extra directory at the destination:
#       ${SCRATCH_HOME}/project_name/data/input/input
# * for more about the (endless) rsync options, see the docs:
#       https://download.samba.org/pub/rsync/rsync.html
# ==============================
# Finally, run the experiment!
# ==============================
# Read line number ${SLURM_ARRAY_TASK_ID} from the experiment file and run it
# ${SLURM_ARRAY_TASK_ID} is simply the number of the job within the array. If
# you execute `sbatch --array=1:100 ...` the jobs will get numbers 1 to 100
# inclusive.

experiment_text_file=$1
COMMAND="`sed \"${SLURM_ARRAY_TASK_ID}q;d\" ${experiment_text_file}`"
echo "Running provided command: ${COMMAND}"
eval "${COMMAND}"
echo "Command ran successfully!"


# ======================================
# Move output data from scratch to DFS
# ======================================
# This presumes your command wrote data to some known directory. In this
# example, send it back to the DFS with rsync

echo "Moving output data back to DFS"

export START_INDEX=$(echo $COMMAND | awk -F'--start_index ' '{print $2}' | awk '{print $1}')
export START_INDEX=$(printf "%05d" $START_INDEX)
export DATASET_SIZE=$(echo $COMMAND | awk -F'--dataset_size_limit ' '{print $2}' | awk '{print $1}')
export DATASET_SIZE=$(printf "%05d" $DATASET_SIZE)

#move results
src_path=${SCRATCH_HOME}/memitpp/results/combined/${MODEL_NAME}/run_${START_INDEX}_${DATASET_SIZE}
dest_path=${repo_home}/results/combined/${MODEL_NAME}/run_${START_INDEX}_${DATASET_SIZE}
mkdir -p ${dest_path}  # make it if required
echo "Moving data from ${src_path} to ${dest_path}"
rsync --archive --update --compress --progress --verbose --log-file=/dev/stdout ${src_path}/ ${dest_path} 

# =========================
# Post experiment logging
# =========================
echo ""
echo "============"
echo "job finished successfully"
dt=$(date '+%d/%m/%Y %H:%M:%S')
echo "Job finished: $dt"