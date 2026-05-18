#!/bin/bash
#SBATCH -J mora-gm-loc
##SBATCH -A your_account
#SBATCH -p ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err

set -eo pipefail

REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
CONDA_ENV="${CONDA_ENV:-hvp_vila2}"
MASTER_PORT="${MASTER_PORT:-25001}"
LOG_BASE_DIR="${LOG_BASE_DIR:-${REPO_DIR}/experiments}"
DATASET_DIR="${DATASET_DIR:-${REPO_DIR}/dataset}"
SAM_CKPT="${SAM_CKPT:-${REPO_DIR}/checkpoints/sam_vit_h_4b8939.pth}"
BASE_MODEL="${BASE_MODEL:-}"
LOC_MODE="${LOC_MODE:-global}"
EXP_NAME="${EXP_NAME:-mora-lisa7b-f5-epoch020-groundmore-loc-${LOC_MODE}}"
NUM_FRAMES="${NUM_FRAMES:-5}"
EPOCHS="${EPOCHS:-20}"
STEPS_PER_EPOCH="${STEPS_PER_EPOCH:-500}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-10}"
WORKERS="${WORKERS:-8}"
LR="${LR:-0.0001}"
LOC_LOSS_WEIGHT="${LOC_LOSS_WEIGHT:-1.0}"
HF_HOME="${HF_HOME:-${REPO_DIR}/.cache/huggingface}"
TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-${REPO_DIR}/.cache/torch_extensions}"

mkdir -p "${LOG_BASE_DIR}/slurm" "${HF_HOME}" "${TORCH_EXTENSIONS_DIR}"

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate "${CONDA_ENV}"
elif [[ -n "${CONDA_SH:-}" && -f "${CONDA_SH}" ]]; then
  source "${CONDA_SH}"
  conda activate "${CONDA_ENV}"
else
  echo "Set CONDA_SH to your conda.sh path or make conda available on PATH." >&2
  exit 1
fi
set -u

cd "${REPO_DIR}"

export HF_HOME
export TRANSFORMERS_CACHE="${HF_HOME}"
export TORCH_EXTENSIONS_DIR
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export NCCL_DEBUG=WARN
export CUDA_DEVICE_MAX_CONNECTIONS=1

echo "Job ID: ${SLURM_JOB_ID:-none}"
echo "Node: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "BASE_MODEL=${BASE_MODEL}"
echo "EXP_NAME=${EXP_NAME}"
echo "NUM_FRAMES=${NUM_FRAMES}"
echo "LOC_MODE=${LOC_MODE}"
echo "WORKERS=${WORKERS}"

if [[ -z "${BASE_MODEL}" ]]; then
  echo "Set BASE_MODEL to a stage-1 LISA/MoRA HF model directory or model id." >&2
  exit 1
fi

test -d "${BASE_MODEL}"
test -f "${DATASET_DIR}/GroundMoRe/trainval_v2.json"
test -d "${DATASET_DIR}/GroundMoRe/groundmore_videos"
test -f "${SAM_CKPT}"

deepspeed --master_port="${MASTER_PORT}" --num_gpus=2 train_ds.py \
  --version="${BASE_MODEL}" \
  --dataset_dir="${DATASET_DIR}" \
  --vision_pretrained="${SAM_CKPT}" \
  --dataset="groundmore" \
  --sample_rates="1" \
  --conv_type="llava_llama_2" \
  --exp_name="${EXP_NAME}" \
  --log_base_dir="${LOG_BASE_DIR}" \
  --steps_per_epoch "${STEPS_PER_EPOCH}" \
  --epochs "${EPOCHS}" \
  --batch_size "${BATCH_SIZE}" \
  --grad_accumulation_steps "${GRAD_ACCUM}" \
  --num_frames "${NUM_FRAMES}" \
  --workers "${WORKERS}" \
  --lr "${LR}" \
  --loc_mode "${LOC_MODE}" \
  --loc_loss_weight "${LOC_LOSS_WEIGHT}" \
  --visualize_freq 100 \
  --keep_ckpt_every 5 \
  --auto_resume
