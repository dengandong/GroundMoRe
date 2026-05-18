#!/bin/bash
#SBATCH -J mora-zs-seg
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
MASTER_PORT="${MASTER_PORT:-24999}"
EXP_NAME="${EXP_NAME:-mora-lisa7b-zs-training}"
LOG_BASE_DIR="${LOG_BASE_DIR:-${REPO_DIR}/experiments}"
DATASET_DIR="${DATASET_DIR:-${REPO_DIR}/dataset}"
SAM_CKPT="${SAM_CKPT:-${REPO_DIR}/checkpoints/sam_vit_h_4b8939.pth}"
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
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export NCCL_DEBUG=WARN
export CUDA_DEVICE_MAX_CONNECTIONS=1

echo "Job ID: ${SLURM_JOB_ID:-none}"
echo "Node: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "Conda env: ${CONDA_ENV}"
which python
which deepspeed

python - <<'PY'
import importlib
for name in ["torch", "transformers", "deepspeed", "cv2", "pycocotools"]:
    importlib.import_module(name)
    print(f"{name}: ok")
PY

test -d "${DATASET_DIR}/refytvos/train/JPEGImages"
test -f "${DATASET_DIR}/refytvos/meta_expressions/train/meta_expressions.json"
test -d "${DATASET_DIR}/MeViSv2/train/JPEGImages"
test -f "${DATASET_DIR}/MeViSv2/train/meta_expressions.json"
test -f "${SAM_CKPT}"

deepspeed --master_port="${MASTER_PORT}" --num_gpus=2 train_ds.py \
  --version="xinlai/LISA-7B-v1" \
  --dataset_dir="${DATASET_DIR}" \
  --vision_pretrained="${SAM_CKPT}" \
  --dataset="refer_video_seg||mevis" \
  --sample_rates="1,1" \
  --conv_type="llava_llama_2" \
  --exp_name="${EXP_NAME}" \
  --log_base_dir="${LOG_BASE_DIR}" \
  --steps_per_epoch 500 \
  --epochs 20 \
  --workers 8 \
  --visualize_freq 100 \
  --seg_only \
  --auto_resume
