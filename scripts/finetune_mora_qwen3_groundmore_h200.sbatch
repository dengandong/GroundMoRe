#!/bin/bash
#SBATCH -J mora-qwen-gm
##SBATCH -A your_account
#SBATCH -p ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=384G
#SBATCH --time=12:00:00
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err

set -eo pipefail

REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
CONDA_ENV="${CONDA_ENV:-hvp_vila2}"
VENV_ACTIVATE="${VENV_ACTIVATE:-}"
MASTER_PORT="${MASTER_PORT:-25071}"
EXP_NAME="${EXP_NAME:-mora-qwen3-vl-8b-epoch005-groundmore-finetune-f5-global-loc}"
LOG_BASE_DIR="${LOG_BASE_DIR:-${REPO_DIR}/experiments}"
DATASET_DIR="${DATASET_DIR:-${REPO_DIR}/dataset}"
SAM_CKPT="${SAM_CKPT:-${REPO_DIR}/checkpoints/sam_vit_h_4b8939.pth}"
QWEN_VERSION="${QWEN_VERSION:-Qwen/Qwen3-VL-8B-Instruct}"
INIT_FROM="${INIT_FROM:-}"
HF_HOME="${HF_HOME:-${REPO_DIR}/.cache/huggingface}"
TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-${REPO_DIR}/.cache/torch_extensions}"
GPUS_PER_NODE="${GPUS_PER_NODE:-2}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-flash_attention_2}"
MODEL_MAX_LENGTH="${MODEL_MAX_LENGTH:-4096}"
QWEN_MIN_PIXELS="${QWEN_MIN_PIXELS:-200704}"
QWEN_MAX_PIXELS="${QWEN_MAX_PIXELS:-802816}"
NUM_FRAMES="${NUM_FRAMES:-5}"
STEPS_PER_EPOCH="${STEPS_PER_EPOCH:-500}"
EPOCHS="${EPOCHS:-20}"
WORKERS="${WORKERS:-8}"
VISUALIZE_FREQ="${VISUALIZE_FREQ:-0}"

mkdir -p "${LOG_BASE_DIR}/slurm" "${HF_HOME}" "${TORCH_EXTENSIONS_DIR}"

if [[ -f "${VENV_ACTIVATE}" ]]; then
  source "${VENV_ACTIVATE}"
elif command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate "${CONDA_ENV}"
elif [[ -n "${CONDA_SH:-}" && -f "${CONDA_SH}" ]]; then
  source "${CONDA_SH}"
  conda activate "${CONDA_ENV}"
else
  echo "Set VENV_ACTIVATE or CONDA_SH, or make conda available on PATH." >&2
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
echo "Qwen version: ${QWEN_VERSION}"
echo "Experiment: ${EXP_NAME}"
echo "Init checkpoint: ${INIT_FROM}"
echo "GPUs per node: ${GPUS_PER_NODE}"
echo "Steps per epoch: ${STEPS_PER_EPOCH}"
echo "Epochs: ${EPOCHS}"
which python
which deepspeed

if [[ -z "${INIT_FROM}" ]]; then
  echo "Set INIT_FROM to a stage-1 MoRA-Qwen DeepSpeed checkpoint directory." >&2
  exit 1
fi

python - <<'PY'
import importlib
import transformers
for name in ["torch", "transformers", "deepspeed", "cv2", "pycocotools", "skimage"]:
    importlib.import_module(name)
    print(f"{name}: ok")
print("transformers:", transformers.__version__)
if not hasattr(transformers, "Qwen3VLForConditionalGeneration"):
    raise RuntimeError("Qwen3VLForConditionalGeneration is unavailable.")
PY

test -f "${DATASET_DIR}/GroundMoRe/trainval_v2.json"
test -d "${DATASET_DIR}/GroundMoRe/groundmore_videos"
test -f "${SAM_CKPT}"
test -f "${INIT_FROM}/latest"

extra_args=()
if [[ -d "${LOG_BASE_DIR}/${EXP_NAME}/ckpt_model" ]]; then
  extra_args+=(--auto_resume)
else
  extra_args+=(--init_from "${INIT_FROM}")
fi

deepspeed --master_port="${MASTER_PORT}" --num_gpus="${GPUS_PER_NODE}" train_qwen3_ds.py \
  --version="${QWEN_VERSION}" \
  --dataset_dir="${DATASET_DIR}" \
  --vision_pretrained="${SAM_CKPT}" \
  --dataset="groundmore" \
  --sample_rates="1" \
  --exp_name="${EXP_NAME}" \
  --log_base_dir="${LOG_BASE_DIR}" \
  --attn_implementation="${ATTN_IMPLEMENTATION}" \
  --model_max_length "${MODEL_MAX_LENGTH}" \
  --qwen_min_pixels "${QWEN_MIN_PIXELS}" \
  --qwen_max_pixels "${QWEN_MAX_PIXELS}" \
  --num_frames "${NUM_FRAMES}" \
  --steps_per_epoch "${STEPS_PER_EPOCH}" \
  --epochs "${EPOCHS}" \
  --workers "${WORKERS}" \
  --visualize_freq "${VISUALIZE_FREQ}" \
  --loc_mode global \
  --keep_ckpt_every 5 \
  "${extra_args[@]}"
