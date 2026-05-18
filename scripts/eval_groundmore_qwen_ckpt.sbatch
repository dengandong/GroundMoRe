#!/bin/bash
#SBATCH -J mora-qwen-eval
##SBATCH -A your_account
#SBATCH -p ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err

set -eo pipefail

REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
CONDA_ENV="${CONDA_ENV:-hvp_vila2}"
VENV_ACTIVATE="${VENV_ACTIVATE:-}"
LOG_BASE_DIR="${LOG_BASE_DIR:-${REPO_DIR}/experiments}"
SAM_CKPT="${SAM_CKPT:-${REPO_DIR}/checkpoints/sam_vit_h_4b8939.pth}"
QWEN_VERSION="${QWEN_VERSION:-Qwen/Qwen3-VL-8B-Instruct}"
CKPT_DIR="${CKPT_DIR:-}"
EVAL_NAME="${EVAL_NAME:-$(basename "$(dirname "${CKPT_DIR}")")-$(basename "${CKPT_DIR}")-f20-loc}"
FP32_STATE="${FP32_STATE:-${LOG_BASE_DIR}/exported/${EVAL_NAME}/fp32_model_state.pt}"
RESULT_FILE="${RESULT_FILE:-${LOG_BASE_DIR}/eval_results/${EVAL_NAME}.txt}"
OUTPUT_DIR="${OUTPUT_DIR:-${LOG_BASE_DIR}/eval_outputs/${EVAL_NAME}}"
NUM_FRAMES="${NUM_FRAMES:-20}"
LOC_MODE="${LOC_MODE:-global}"
LOC_THRESHOLD="${LOC_THRESHOLD:-0.5}"
MAX_VIDEOS="${MAX_VIDEOS:-}"
MAX_EXPRESSIONS="${MAX_EXPRESSIONS:-}"
RESUME_EVAL="${RESUME_EVAL:-1}"
READ_RETRIES="${READ_RETRIES:-5}"

mkdir -p "$(dirname "${FP32_STATE}")" "$(dirname "${RESULT_FILE}")" "${OUTPUT_DIR}" "${LOG_BASE_DIR}/slurm"

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
cd "${REPO_DIR}"

export HF_HOME="${HF_HOME:-${REPO_DIR}/.cache/huggingface}"
export TRANSFORMERS_CACHE="${HF_HOME}"
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

echo "Job ID: ${SLURM_JOB_ID:-none}"
echo "Node: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "QWEN_VERSION=${QWEN_VERSION}"
echo "CKPT_DIR=${CKPT_DIR}"
echo "FP32_STATE=${FP32_STATE}"
echo "RESULT_FILE=${RESULT_FILE}"
echo "NUM_FRAMES=${NUM_FRAMES}"
echo "LOC_MODE=${LOC_MODE}"
echo "LOC_THRESHOLD=${LOC_THRESHOLD}"
echo "RESUME_EVAL=${RESUME_EVAL}"

if [[ -z "${CKPT_DIR}" ]]; then
  echo "Set CKPT_DIR to a DeepSpeed checkpoint directory." >&2
  exit 1
fi

test -f "${CKPT_DIR}/latest"
test -f "${CKPT_DIR}/zero_to_fp32.py"
test -f "${SAM_CKPT}"
test -f "groundmore_test.json"
test -d "groundmore_videos"

if [[ ! -f "${FP32_STATE}" && ! -d "${FP32_STATE}" ]]; then
  python "${CKPT_DIR}/zero_to_fp32.py" "${CKPT_DIR}" "${FP32_STATE}"
fi

eval_args=(
  --version "${QWEN_VERSION}"
  --weight "${FP32_STATE}"
  --vision_pretrained "${SAM_CKPT}"
  --video_root "groundmore_videos"
  --meta_file "groundmore_test.json"
  --output_dir "${OUTPUT_DIR}"
  --result_file "${RESULT_FILE}"
  --num_frames "${NUM_FRAMES}"
  --loc_mode "${LOC_MODE}"
  --loc_threshold "${LOC_THRESHOLD}"
  --read_retries "${READ_RETRIES}"
)
if [[ "${RESUME_EVAL}" == "1" ]]; then
  eval_args+=(--resume)
else
  eval_args+=(--overwrite)
fi
if [[ -n "${MAX_VIDEOS}" ]]; then
  eval_args+=(--max_videos "${MAX_VIDEOS}")
fi
if [[ -n "${MAX_EXPRESSIONS}" ]]; then
  eval_args+=(--max_expressions "${MAX_EXPRESSIONS}")
fi

python evaluate_groundmore_qwen.py "${eval_args[@]}"
