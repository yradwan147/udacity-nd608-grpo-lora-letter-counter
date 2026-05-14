#!/bin/bash --login
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --constraint=a100
#SBATCH --partition=batch
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=64G
#SBATCH -J nd608_p1_grpo
#SBATCH -o /home/radwany/UdacityStuff/work/project1/slurm_logs/grpo_%J.out
#SBATCH -e /home/radwany/UdacityStuff/work/project1/slurm_logs/grpo_%J.err

# Udacity nd608 Project 1: GRPO + LoRA on Qwen2.5-3B-Instruct.

set -euo pipefail

mkdir -p /home/radwany/UdacityStuff/work/project1/slurm_logs

PROJECT_ROOT="/home/radwany/UdacityStuff/cd13303-genai-c1-classroom/project/starter"
VENV="/home/radwany/UdacityStuff/work/project1/venv"

cd "${PROJECT_ROOT}"

# Activate the pre-installed venv (created on the login node).
source "${VENV}/bin/activate"
export PATH="${VENV}/bin:${PATH}"
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
# Force HuggingFace to download to a writable location.
export HF_HOME="/home/radwany/UdacityStuff/work/project1/hf_cache"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
mkdir -p "${HF_HOME}" "${TRANSFORMERS_CACHE}"

echo "Job ID:    ${SLURM_JOB_ID:-N/A}"
echo "Node:      ${SLURM_NODELIST:-N/A}"
echo "GPU info:  $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "Python:    $(python --version) at $(which python)"
echo "Start:     $(date)"

# Run the notebook end-to-end.
python -m jupyter nbconvert \
  --to notebook \
  --execute gen_ai_fundamentals_project_starter.ipynb \
  --output executed_gen_ai_fundamentals_project_starter.ipynb \
  --ExecutePreprocessor.timeout=10800 \
  --ExecutePreprocessor.kernel_name=python3

echo "Notebook executed. Now packaging results."

# Copy results into deliverables/project1 for later zipping.
DELIV="/home/radwany/UdacityStuff/deliverables/project1"
mkdir -p "${DELIV}"
cp -v executed_gen_ai_fundamentals_project_starter.ipynb "${DELIV}/"
cp -v gen_ai_fundamentals_project_starter.ipynb "${DELIV}/"
[ -d grpo_saved_lora ] && cp -rv grpo_saved_lora "${DELIV}/"
cp -v /home/radwany/UdacityStuff/cd13303-genai-c1-classroom/project/requirements.txt "${DELIV}/"

echo "End time: $(date)"
