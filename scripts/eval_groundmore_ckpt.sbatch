#!/bin/bash
#SBATCH -J mora-eval
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
LOG_BASE_DIR="${LOG_BASE_DIR:-${REPO_DIR}/experiments}"
DATASET_DIR="${DATASET_DIR:-${REPO_DIR}/dataset}"
SAM_CKPT="${SAM_CKPT:-${REPO_DIR}/checkpoints/sam_vit_h_4b8939.pth}"
BASE_MODEL="${BASE_MODEL:-xinlai/LISA-7B-v1}"
CKPT_DIR="${CKPT_DIR:-}"
EVAL_NAME="${EVAL_NAME:-$(basename "$(dirname "${CKPT_DIR}")")-$(basename "${CKPT_DIR}")}"
EXPORT_DIR="${EXPORT_DIR:-${LOG_BASE_DIR}/exported/${EVAL_NAME}}"
FP32_STATE="${FP32_STATE:-${EXPORT_DIR}/fp32_model_state.pt}"
RESULT_FILE="${RESULT_FILE:-${LOG_BASE_DIR}/eval_results/${EVAL_NAME}.txt}"
OUTPUT_DIR="${OUTPUT_DIR:-${LOG_BASE_DIR}/eval_outputs/${EVAL_NAME}}"
NUM_FRAMES="${NUM_FRAMES:-20}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32}"
FORCE_SEG_TOKEN="${FORCE_SEG_TOKEN:-0}"
FORCE_LOC_TOKEN="${FORCE_LOC_TOKEN:-0}"
USE_LOC_REFINE="${USE_LOC_REFINE:-0}"
LOC_THRESHOLD="${LOC_THRESHOLD:-0.5}"
SEG_ONLY="${SEG_ONLY:-1}"
LOC_MODE="${LOC_MODE:-global}"
MAX_VIDEOS="${MAX_VIDEOS:-}"
MAX_EXPRESSIONS="${MAX_EXPRESSIONS:-}"
RESUME_EVAL="${RESUME_EVAL:-0}"
READ_RETRIES="${READ_RETRIES:-3}"

mkdir -p "${EXPORT_DIR}" "$(dirname "${RESULT_FILE}")" "${OUTPUT_DIR}" "${LOG_BASE_DIR}/slurm"

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

export HF_HOME="${HF_HOME:-${REPO_DIR}/.cache/huggingface}"
export TRANSFORMERS_CACHE="${HF_HOME}"
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

echo "Job ID: ${SLURM_JOB_ID:-none}"
echo "Node: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "CKPT_DIR=${CKPT_DIR}"
echo "EXPORT_DIR=${EXPORT_DIR}"
echo "RESULT_FILE=${RESULT_FILE}"
echo "SEG_ONLY=${SEG_ONLY}"
echo "LOC_MODE=${LOC_MODE}"
echo "FORCE_SEG_TOKEN=${FORCE_SEG_TOKEN}"
echo "FORCE_LOC_TOKEN=${FORCE_LOC_TOKEN}"
echo "USE_LOC_REFINE=${USE_LOC_REFINE}"
echo "LOC_THRESHOLD=${LOC_THRESHOLD}"
echo "RESUME_EVAL=${RESUME_EVAL}"
echo "READ_RETRIES=${READ_RETRIES}"

if [[ -z "${CKPT_DIR}" ]]; then
  echo "Set CKPT_DIR to a DeepSpeed checkpoint directory." >&2
  exit 1
fi

test -f "${CKPT_DIR}/latest"
test -f "${CKPT_DIR}/zero_to_fp32.py"
test -f "${SAM_CKPT}"
test -f "groundmore_test.json"
test -d "groundmore_videos"

if [[ ! -f "${FP32_STATE}" ]]; then
  python "${CKPT_DIR}/zero_to_fp32.py" "${CKPT_DIR}" "${FP32_STATE}"
fi

if [[ ! -f "${EXPORT_DIR}/config.json" ]]; then
  merge_args=(
    --version="${BASE_MODEL}" \
    --vision_pretrained="${SAM_CKPT}" \
    --weight="${FP32_STATE}" \
    --save_path="${EXPORT_DIR}" \
    --conv_type="llava_llama_2" \
    --loc_mode="${LOC_MODE}"
  )
  if [[ "${SEG_ONLY}" == "1" ]]; then
    merge_args+=(--seg_only)
  fi
  python merge_lora_weights_and_save_hf_model.py "${merge_args[@]}"
fi

eval_args=(
  --version "${EXPORT_DIR}"
  --vision_pretrained "${SAM_CKPT}"
  --conv_type "llava_llama_2"
  --video_root "groundmore_videos"
  --meta_file "groundmore_test.json"
  --output_dir "${OUTPUT_DIR}"
  --result_file "${RESULT_FILE}"
  --num_frames "${NUM_FRAMES}"
  --max_new_tokens "${MAX_NEW_TOKENS}"
  --read_retries "${READ_RETRIES}"
)
if [[ "${RESUME_EVAL}" == "1" ]]; then
  eval_args+=(--resume)
else
  eval_args+=(--overwrite)
fi
if [[ "${SEG_ONLY}" == "1" ]]; then
  eval_args+=(--seg_only)
fi

if [[ -n "${MAX_VIDEOS}" ]]; then
  eval_args+=(--max_videos "${MAX_VIDEOS}")
fi
if [[ -n "${MAX_EXPRESSIONS}" ]]; then
  eval_args+=(--max_expressions "${MAX_EXPRESSIONS}")
fi
if [[ "${FORCE_SEG_TOKEN}" == "1" ]]; then
  eval_args+=(--force_seg_token)
fi
if [[ "${FORCE_LOC_TOKEN}" == "1" ]]; then
  eval_args+=(--force_loc_token)
fi
if [[ "${USE_LOC_REFINE}" == "1" ]]; then
  eval_args+=(--use_loc_refine --loc_threshold "${LOC_THRESHOLD}")
fi

python evaluate_groundmore.py "${eval_args[@]}"
