# Example Usage
# DATA_DIR="/home/ubuntu/porialab-us-midwest-1/Tej/mmr-eval/traces_data_backup"

# source /scratch_aisg/SPEC-SF-AISG/ob1/mmr-eval/qwen-evaluation/.venv/bin/activate

# Add the parent directory to PYTHONPATH so imports work correctly
export PYTHONPATH="${PYTHONPATH}:/home/ubuntu/poria-cvpr-2026/ob1/vlprm/"

OUTPUT_DATA_DIR="/home/ubuntu/poria-cvpr-2026/ob1/vlprm/eval/tts_eval/step_aggregation_search/outputs"
# BASE_DATA_DIR="/home/ubuntu/poria-cvpr-2026/ob1/vlprm/eval/tts_eval/reward_guided_search/VisualPRM_relabelling"
INPUT_JSON_DATA_PATH="/home/ubuntu/poria-cvpr-2026/Tej/mmr-eval/traces_data/g12b_policy_step/q3b_prm/Q3B_mc0_sr_mc0_full_bs2_gs4_lr1e-5_VF_0827_1452_puzzleVQA_1K_subset_result-merged-0-1000-20250902_232205.json"

CHECKPOINT_BASE_PATH="/home/ubuntu/poria-cvpr-2026/ob1/CVPR_PRM/src/training/trained_models"
# MODEL_PATH="${CHECKPOINT_BASE_PATH}/ob11_Qwen-VL-PRM-7B_grpo_listwise_ce_20251106_223500"
MODEL_PATH="${CHECKPOINT_BASE_PATH}/ob11_Qwen-VL-PRM-7B_grpo_listwise_ce_20251106_215426"
# MODEL_PATH="ob11/Qwen-VL-PRM-3B"
# MODEL_PATH="OpenGVLab/VisualPRM-8B-v1_1"

# Extract datetime from MODEL_PATH if it contains YYYYMMDD_HHMMSS pattern for use in model_prefix
if [[ $MODEL_PATH =~ ([0-9]{8}_[0-9]{6})$ ]]; then
    model_datetime=$(echo "${BASH_REMATCH[1]}" | tr '_' '-')
else
    model_datetime=""  # No datetime suffix if not found in MODEL_PATH
fi

# Generate current datetime with Singapore timezone for output paths and logs
export TZ=Asia/Singapore
current_datetime=$(date +"%Y%m%d-%H%M%S")

# Infer dataset shorthand from INPUT_JSON_DATA_PATH
if [[ $INPUT_JSON_DATA_PATH =~ mathvista_testmini ]]; then
    dataset_shorthand="m-vista"
elif [[ $INPUT_JSON_DATA_PATH =~ MMMU_DEV_VAL ]]; then
    dataset_shorthand="MMMU"
elif [[ $INPUT_JSON_DATA_PATH =~ puzzleVQA_1K_subset ]]; then
    dataset_shorthand="puzzle"
elif [[ $INPUT_JSON_DATA_PATH =~ AlgoPuzzleVQA_900_subset ]]; then
    dataset_shorthand="algopuz"
elif [[ $INPUT_JSON_DATA_PATH =~ mathvision_test ]]; then
    dataset_shorthand="m-vision"
else
    dataset_shorthand="unknown"  # fallback
fi

# Determine model prefix and add suffix based on whether it's finetuned or base
# Include datetime from MODEL_PATH in model_prefix if available
if [[ $MODEL_PATH =~ [V]isualPRM-8B ]]; then
    reward_model_prefix="V8B"
elif [[ $MODEL_PATH =~ [V]isualPRM-8B-v1_1 ]]; then
    reward_model_prefix="V8B-v1_1"
elif [[ $MODEL_PATH =~ [Q]wen-VL-PRM-3B ]]; then
    if [[ $MODEL_PATH == ${CHECKPOINT_BASE_PATH}* ]]; then
        reward_model_prefix="VL-3B-ft"
    else
        reward_model_prefix="VL-3B-base"
    fi
elif [[ $MODEL_PATH =~ [Q]wen-VL-PRM-7B ]]; then
    if [[ $MODEL_PATH == ${CHECKPOINT_BASE_PATH}* ]]; then
        reward_model_prefix="VL-7B-ft"
    else
        reward_model_prefix="VL-7B-base"
    fi
else
    reward_model_prefix="UNKNOWN"  # fallback
fi

POLICY_MODEL="g12b_policy_step"

# Append datetime from MODEL_PATH to model_prefix if available
if [[ -n "$model_datetime" ]]; then
    reward_model_prefix="${reward_model_prefix}_${model_datetime}"
fi

# RUN_SETTING="step_agg"
RUN_SETTING="non_greedy"

base_job_name_prefix="PRM_${reward_model_prefix}"

mkdir -p logs/inference_logs/${POLICY_MODEL}/prm_${reward_model_prefix}

CUDA_VISIBLE_DEVICES=7 python prm_tts_eval.py \
    --model-path $MODEL_PATH \
    --data-path ${INPUT_JSON_DATA_PATH} \
    --output-path ${OUTPUT_DATA_DIR}/${POLICY_MODEL}/prm_${reward_model_prefix}/${dataset_shorthand}_${RUN_SETTING}-${current_datetime}.json \
    --tts-type ${RUN_SETTING} \
    2>&1 | tee logs/inference_logs/${POLICY_MODEL}/prm_${reward_model_prefix}/${dataset_shorthand}_${RUN_SETTING}-${current_datetime}
