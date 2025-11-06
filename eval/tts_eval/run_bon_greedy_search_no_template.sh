#!/bin/bash

# Add the parent directory to PYTHONPATH so imports work correctly
export PYTHONPATH="${PYTHONPATH}:/home/ubuntu/poria-cvpr-2026/ob1/vlprm/"

# source /home/ubuntu/poria-cvpr-2026/ob1/vlprm/eval/.venv/bin/activate
echo "Python path after activation: $(which python)"
echo "Python version: $(python --version)"

cd reward_guided_search/

# dataset_list=("mmmu_dev" "mathvista_testmini")
dataset_list=(
    # "mathvista_testmini"
    "MMMU_DEV_VAL" 
    # "puzzleVQA_1K_subset" 
    # "AlgoPuzzleVQA_900_subset"
    # "mathvision_test"
)

# POLICY_MODEL_PATH="Qwen/Qwen2.5-VL-3B-Instruct"
POLICY_MODEL_PATH="google/gemma-3-12b-it"
# POLICY_MODEL_PATH="openbmb/MiniCPM-V-2_6" # 8B
# POLICY_MODEL_PATH="OpenGVLab/InternVL2_5-8B"

# CHECKPOINT_BASE_PATH="/data/projects/71001002/ob1/prm-training-code/training_outputs"
# checkpoint="${CHECKPOINT_BASE_PATH}/Q3B_mc0_sr_mc0_balanced_bs2_gs4_lr1e-5_VF_0827_1452"
checkpoint="/home/ubuntu/poria-cvpr-2026/ob1/CVPR_PRM/src/training/trained_models/ob11_Qwen-VL-PRM-3B_grpo_listwise_ce_20251106_210950"

export OPENAI_API_KEY="sk-proj-1234567890" #dummy for now

# Performing Search and Generating Responses
for dataset in "${dataset_list[@]}"; do
    echo "Dataset: $dataset"
    CUDA_VISIBLE_DEVICES=0,1 python vllm_bon_greedy_search_no_template.py \
        --policy_model_path $POLICY_MODEL_PATH \
        --reward_model_path $checkpoint \
        --data $dataset \
        --output_dir ./outputs/DEBUG/$dataset-results \
        --development_mode
done


# Evaluating Responses

# python collate_final_eval_results.py path_to_json_result_file.json