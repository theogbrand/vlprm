import math
import os
import argparse
import json
import datetime
import random
import torch
import glob
import pandas as pd
import numpy as np
import string
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoProcessor, LlavaNextProcessor, AutoTokenizer, Gemma3Processor, AutoModel
from VisualPRMv2 import VisualPRM
from prompts import (
    POLICY_VISUAL_ANALYST_SYS_PROMPT_V3,
)
from utils.utils import (
    sample_to_images_list,
    convert_images_to_base64,
    prepare_question_array_with_base64_image_strings,
    prepare_question_array_with_base64_image_strings_for_internvl_and_minicpm,
    prepare_question_array_with_base64_image_strings_mathvision,
    extract_boxed,
)
from puzzleTest_helpers import (
    load_puzzle_subset, build_puzzlevqa_prompt, build_algopuzzlevqa_prompt, get_puzzle_dataset_info, PUZZLE_DATASET_TYPES, ALGOPUZZLE_DATASET_TYPES, build_puzzlevqa_prompt_minicpm, build_algopuzzlevqa_prompt_minicpm
)
from eval.eval_datasets import PUZZLE_VQA_DATA, ALGO_PUZZLE_VQA_DATA
import regex

from vllm import LLM, SamplingParams
from PIL import Image
import base64
from io import BytesIO

# from qwenvl_utils import process_vision_info
from utils.logger import log_info

import json
# import ast
import traceback

# from evaluation.common.send_telegram_notifications_helper import (
#     send_telegram_error_notification,
# )
from eval.tts_eval.reward_guided_search.collate_final_eval_results import calculate_evaluation_score_direct 
from eval.tts_eval.reward_guided_search.mathvista_helper_functions import (
    load_mathvista_dataset,
    build_mathvista_prompt,
    build_minicpm_mathvista_cot_prompt,
    load_mathvista_dataset_from_file,
)

from eval.tts_eval.reward_guided_search.mathvision_helper_functions import (
    load_mathvision_dataset,
    build_mathvision_prompt,
)

NEWLINE_STEP_SEPERATOR = "\n\n"
BOXED_ANSWER_STR = r"\boxed{"

def is_valid_step(step: str) -> bool:
    allowed_pattern = regex.compile(
        r"[^\p{Latin}\p{Greek}\d\s"
        r"\+\-\*/=\^\_\'\".,:;?!\(\)\{\}\[\]\\\$%<>|&@#"
        r"√∞±×÷°]"
    )
    return len(allowed_pattern.findall(step)) > 0

def toliststr(s):
    """Convert string representation of list to actual list"""
    if isinstance(s, str):
        try:
            import ast
            return ast.literal_eval(s)
        except (ValueError, SyntaxError):
            return [s]
    return s if isinstance(s, list) else [s]

def load_mmmu_dataset(data_dir, dataset_name='MMMU_DEV_VAL'):
    """Load the MMMU dataset from TSV file."""
    data_path = os.path.join(data_dir, f"{dataset_name}.tsv")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset file not found: {data_path}")
    
    # Load the dataset
    data = pd.read_csv(data_path, sep="\t")
    data = data[data["split"] != "dev"]
    
    # Process the dataset
    data['index'] = [str(x) for x in data['index']]
    
    # Handle image data
    if 'image' in data:
        data['image'] = [str(x) for x in data['image']]
        image_map = {x: y for x, y in zip(data['index'], data['image'])}
        for k in image_map:
            if len(image_map[k]) <= 64:
                idx = image_map[k]
                assert idx in image_map and len(image_map[idx]) > 64
                image_map[k] = image_map[idx]

        images = [toliststr(image_map[k]) for k in data['index']]
        data['image'] = [x[0] if len(x) == 1 else x for x in images]
    
    # Handle image paths
    if 'image_path' in data:
        paths = [toliststr(x) for x in data['image_path']]
        data['image_path'] = [x[0] if len(x) == 1 else x for x in paths]
    
    # Convert index to int if possible
    if np.all([isinstance(x, int) or x.isdigit() for x in data['index']]):
        data['index'] = [int(x) for x in data['index']]
    
    return data

os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'

def get_next_step(
    policy_model, image_data_base64_list, previous_steps, policy_prompt, eos_token, tokenizer=None, policy_prompt_array=[]
):
    # to perform assistant pre-filling if previous steps exist
    previous_steps_pre_filling = (
        "" if len(previous_steps) == 0 else "\n\n".join(previous_steps)
    )
    log_info(f"Previous steps pre-filling: {previous_steps_pre_filling}")
    
    if len(previous_steps) > 0:
        log_info(
            f"Appending previous steps to policy prompt: {previous_steps_pre_filling}"
        )
        now_prompt = policy_prompt + "\n\n" + previous_steps_pre_filling
    else:
        now_prompt = policy_prompt

    if (
        hasattr(tokenizer, "name_or_path")
        and tokenizer.name_or_path == "openbmb/MiniCPM-V-2_6"
    ):
        policy_model_name = "MiniCPM"
    else: 
        model_config = policy_model.llm_engine.get_model_config()
        policy_model_name = model_config.model
    # VLLM sampling parameters
    if "qwen" in policy_model_name.lower():
        sampling_params = SamplingParams(
            n=16,
            top_p=0.8,
            top_k=20,
            temperature=0.7,
            max_tokens=16384, # 8192 for BON step-by-step, 16384 for NG step-by-step
            repetition_penalty=1.05, # https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/blob/main/generation_config.json
            # stop=[eos_token, NEWLINE_STEP_SEPERATOR],
            # include_stop_str_in_output=True,
        )
    elif "gemma" in policy_model_name.lower(): # https://docs.unsloth.ai/basics/gemma-3-how-to-run-and-fine-tune
        sampling_params = SamplingParams(
            n=16,
            top_p=0.95,
            top_k=64,
            temperature=0.7,
            max_tokens=16384,
            # stop=[eos_token, NEWLINE_STEP_SEPERATOR],  # BON Greedy is now also working with NG
            # include_stop_str_in_output=True,
        )
    elif "minicpm" in policy_model_name.lower(): 
        sampling_params = SamplingParams(
            n=16,
            temperature=0.0,
            # top_p=0.95,
            # top_k=64,
            # stop=['<|im_end|>', '<|endoftext|>'], # TODO: Important when doing Greedy, not when Non-Greedy, check this again when running
            # include_stop_str_in_output=True,
            max_tokens=16384,
        )
    else:
        raise ValueError(f"Unsupported policy model to set SamplingParams in get_next_step: {policy_model_name}")
    
    log_info(f"Policy Model Sampling parameters: {sampling_params}")
    log_info(f"Sending prompt to policy model: {now_prompt}")
    log_info(f"Length of image data base64 list: {len(image_data_base64_list)}")
    
    # Convert base64 images to PIL Images for VLLM
    images = []
    for base64_img in image_data_base64_list:
        img_data = base64.b64decode(base64_img)
        if "minicpm" in policy_model_name.lower():
            img = Image.open(BytesIO(img_data)).convert('RGB')
            img_width, img_height = img.width, img.height # resizing https://github.com/OpenBMB/MiniCPM-V/blob/main/eval_mm/vlmevalkit/vlmeval/vlm/minicpm_v.py#L260
            if (img_width * img_height) >= (1344 * 1344):
                log_info(f"Image is too large, skipping resize: {img_width}x{img_height}")
                img = img
            else:
                ratio = math.sqrt((1344 * 1344) / (img_width * img_height))
                max_img_width = int(img_width * ratio)
                new_img_width = random.randint(img_width, max_img_width)
                new_img_height = int(new_img_width / img_width * img_height)
                log_info(f"Resizing image from {img_width}x{img_height} to {new_img_width}x{new_img_height}")
                img = img.resize((new_img_width, new_img_height))
        else:
            img = Image.open(BytesIO(img_data))
            img.load()
        images.append(img)
    
    if "minicpm" in policy_model_name.lower():
        # sys_prompt = policy_prompt_array[0]['content']
        sys_prompt = "You are an expert visual analyst who solves complex visual reasoning problems by systematically connecting what you observe in images to the specific requirements of each problem."
        user_prompt = policy_prompt_array[1]['content']
        combined_user_prompt = sys_prompt + "\n" + user_prompt
        now_prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": combined_user_prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        now_prompt = now_prompt.replace("<|im_start|>system\nYou are a helpful assistant.<|im_end|>", "")
        log_info(f"Policy prompt: {now_prompt}")

    # VLLM generation call
    llm_outputs = policy_model.generate(
        {
            "prompt": now_prompt,
            "multi_modal_data": {"image": images} if images else None,
        },
        sampling_params,
    )
    log_info(f"Policy model returned: {llm_outputs}")
    
    # Process VLLM outputs
    candidates = []
    str_set = set()
    for output in llm_outputs[0].outputs:
        gen_text = output.text.replace("\\n", "\n")
        if not gen_text or not gen_text.strip():
            continue
        if gen_text.endswith("\n\n"):
            gen_text = gen_text[:-2].rstrip() + "\n\n"
        else:
            gen_text += (
                eos_token  # signals no longer needs to reason, can extract answer
            )
        if gen_text in str_set:
            continue

        # Create a candidate object with both text and meta_info
        candidate = {
            "text": gen_text, 
            "meta_info": {
                "finish_reason": str(output.finish_reason),
            }
        }
        candidates.append(candidate)
        str_set.add(gen_text)
    log_info(f"New candidates returned from policy model: {len(candidates)}")
    return candidates


# Logprob-based answer selection functions removed - using simplified approach


def check_chat_template_jinja(
    reward_model_path: str, reward_model_architecture: str, model_config_json: dict
) -> None:
    chat_templates = {
        "Qwen2_5_VLForConditionalGeneration": """{% set image_count = namespace(value=0) %}{% set video_count = namespace(value=0) %}{% for message in messages %}{% if loop.first and message['role'] != 'system' %}<|im_start|>system
You are a helpful assistant.<|im_end|>
{% endif %}<|im_start|>{{ message['role'] }}
{% if message['content'] is string %}{{ message['content'] }}<|im_end|>
{% else %}{% for content in message['content'] %}{% if content['type'] == 'image' or 'image' in content or 'image_url' in content %}{% set image_count.value = image_count.value + 1 %}{% if add_vision_id %}Picture {{ image_count.value }}: {% endif %}<|vision_start|><|image_pad|><|vision_end|>{% elif content['type'] == 'video' or 'video' in content %}{% set video_count.value = video_count.value + 1 %}{% if add_vision_id %}Video {{ video_count.value }}: {% endif %}<|vision_start|><|video_pad|><|vision_end|>{% elif 'text' in content %}{{ content['text'] }}{% endif %}{% endfor %}<|im_end|>
{% endif %}{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant
{% endif %}""",
        "LlavaNextForConditionalGeneration": {
            "LlamaForCausalLM": """{% for message in messages %}{{ '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'}}{# Render all images first #}{% for content in message['content'] | selectattr('type', 'equalto', 'image') %}{{ '<image>' }}{% endfor %}{# Render all text next #}{% for content in message['content'] | selectattr('type', 'equalto', 'text') %}{{ '\n' + content['text'] + '<|eot_id|>' }}{% endfor %}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}""",
            "MistralForCausalLM": """{% for message in messages %}{% if message['role'] == 'system' %}{{ '<<SYS>>\n' + message['content'][0]['text'] + '\n<</SYS>>\n\n' }}{% elif message['role'] == 'user' %}{{ '[INST] ' }}{# Render all images first #}{% for content in message['content'] | selectattr('type', 'equalto', 'image') %}{{ '<image>\n' }}{% endfor %}{# Render all text next #}{% for content in message['content'] | selectattr('type', 'equalto', 'text') %}{{ content['text'] }}{% endfor %}{{' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ ' ' + message['content'][0]['text'] + '</s> '}}{% else %}{{ raise_exception('Only user and assistant roles are supported!') }}{% endif %}{% endfor %}""",
        },
    }
    chat_template_path = os.path.join(reward_model_path, "chat_template.jinja")
    chat_template_contents = chat_templates[reward_model_architecture]

    text_llm_backbone = model_config_json["text_config"]["architectures"][0]
    if not os.path.exists(chat_template_path):
        with open(chat_template_path, "w", encoding="utf-8") as f:
            f.write(chat_template_contents)
        log_info(f"Created default chat template at: {chat_template_path}")
    elif (
        text_llm_backbone == "MistralForCausalLM"
        and reward_model_architecture == "LlavaNextForConditionalGeneration"
        and os.path.exists(chat_template_path)
    ):
        # override the existing chat template
        with open(chat_template_path, "w", encoding="utf-8") as f:
            f.write(chat_template_contents["MistralForCausalLM"])
        log_info(
            f"Overridden existing chat template for LlavaNext Mistral at: {chat_template_path}"
        )
    else:
        log_info(f"Using existing chat template at: {chat_template_path} for {reward_model_architecture} with text_llm_backbone {text_llm_backbone}")

def check_preprocessor_config(
    reward_model_path: str, reward_model_architecture: str
) -> None:
    preprocessor_configs = {
        "Qwen2_5_VLForConditionalGeneration": {
            "min_pixels": 3136,
            "max_pixels": 12845056,
            "patch_size": 14,
            "temporal_patch_size": 2,
            "merge_size": 2,
            "image_mean": [0.48145466, 0.4578275, 0.40821073],
            "image_std": [0.26862954, 0.26130258, 0.27577711],
            "image_processor_type": "Qwen2VLImageProcessor",
            "processor_class": "Qwen2_5_VLProcessor",
        },
        "LlavaNextForConditionalGeneration": {
            "aspect_ratio_setting": "anyres",
            "crop_size": {"height": 336, "width": 336},
            "do_center_crop": True,
            "do_convert_rgb": True,
            "do_normalize": True,
            "do_pad": True,
            "do_rescale": True,
            "do_resize": True,
            "image_grid_pinpoints": [
                [336, 672],
                [672, 336],
                [672, 672],
                [1008, 336],
                [336, 1008],
            ],
            "image_mean": [0.48145466, 0.4578275, 0.40821073],
            "image_processor_type": "LlavaNextImageProcessor",
            "image_std": [0.26862954, 0.26130258, 0.27577711],
            "processor_class": "LlavaNextProcessor",
            "resample": 3,
            "rescale_factor": 0.00392156862745098,
            "size": {"shortest_edge": 336},
        },
    }
    try:
        preprocessor_config_path = os.path.join(
            reward_model_path, "preprocessor_config.json"
        )
        if not os.path.exists(preprocessor_config_path):
            with open(preprocessor_config_path, "w") as f:
                json.dump(preprocessor_configs[reward_model_architecture], f, indent=2)
            log_info(
                f"Created default preprocessor config at: {preprocessor_config_path}"
            )
        else:
            log_info(
                f"Using existing preprocessor config at: {preprocessor_config_path}"
            )
    except Exception as e:
        raise ValueError(
            f"Error when checking preprocessor config for {reward_model_architecture}: {e}"
        )

def main():
    parser = argparse.ArgumentParser(description="Greedy Search reasoning pipeline with reward model")

    parser.add_argument("--policy_model_path", type=str, required=True, help="Path to the policy model.")
    parser.add_argument("--reward_model_path", type=str, required=True, help="Path to the reward model.")

    # Build dataset choices - MMMU + PuzzleVQA 1K subset (individual puzzle types still available for dev)
    dataset_choices = [
        "MMMU_DEV_VAL", 
        "puzzleVQA_1K_subset", 
        "AlgoPuzzleVQA_900_subset", 
        "mathvista_testmini",
        "mathvision_test"
        ]
    # Add individual puzzle types for development/testing (commented out for main evaluation)
    # puzzle_choices = [f"puzzle_{puzzle_type}" for puzzle_type in PUZZLE_DATASET_TYPES]
    # dataset_choices.extend(puzzle_choices)
    
    parser.add_argument("--data", type=str, required=True, help="Dataset to Evaluate on", 
                        choices=dataset_choices)

    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the results.")
    parser.add_argument("--temperature", type=float, default=0.7, help="the temperature of the policy model.")
    parser.add_argument(
        "--data_begin",
        type=int,
        default=0,
        help="Starting index of the dataset to process.",
    )
    parser.add_argument(
        "--data_end",
        type=int,
        default=None,
        help="Ending index of the dataset to process. Defaults: mmmu_dev=150, mmmu_validation=900, mmmu_test_full=10500, others=10.",
    )
    parser.add_argument(
        "--development_mode",
        action="store_true",
        default=False,
        help="If True, filter dataset to target_ids after loading.",
    )
    parser.add_argument(
        "--policy_gpu",
        type=int,
        default=0,
        help="GPU device ID for policy model (default: 0)",
    )
    parser.add_argument(
        "--reward_gpu",
        type=int,
        default=1,
        help="GPU device ID for reward model (default: 1)",
    )
    parser.add_argument(
        "--partition_id",
        type=int,
        default=None,
        help="Partition ID for parallel runs (adds partition suffix to output file)",
    )
    parser.add_argument(
        "--run_datetime",
        type=str,
        default=None,
        help="Run datetime string for consistent naming across parallel partitions",
    )
    parser.add_argument(
        "--data_dir", 
        type=str, 
        default="/home/ubuntu/poria-cvpr-2026/ob1/vlprm/eval/eval_datasets/", 
        help="The absolute path of MMMU dataset directory"
    )
    # parser.add_argument(
    #     "--puzzlevqa_data_dir",
    #     type=str,
    #     default="/scratch_aisg/SPEC-SF-AISG/ob1/mmr-eval/LLM-PuzzleTest/PuzzleVQA/data",
    #     help="The absolute path of PuzzleVQA dataset directory"
    # )
    parser.add_argument(
        "--use_cot", 
        action="store_true", 
        help="Use Chain-of-Thought prompting"
    )
    parser.add_argument(
        "--cot_prompt", 
        type=str, 
        default="", 
        help="Custom Chain-of-Thought prompt"
    )

    args = parser.parse_args()
    
    # Set default data_end based on dataset if not specified
    if args.data_end is None:
        if args.data == "MMMU_DEV_VAL":
            args.data_end = 900  # Default for MMMU_DEV_VAL
        elif args.data == "puzzleVQA_1K_subset":
            args.data_end = 1000  # PuzzleVQA 1K subset has 1000 samples (50 × 20 puzzle types)
        elif args.data == "AlgoPuzzleVQA_900_subset":
            args.data_end = 900  # PuzzleVQA 1K subset has 1000 samples (50 × 20 puzzle types)
        # elif args.data.startswith("puzzle_"):
        #     args.data_end = 100  # Individual PuzzleVQA datasets have 100 samples
        elif args.data == "mathvista_testmini":
            args.data_end = 1000
        elif args.data == "mathvision_test":
            args.data_end = 3040
        else:
            raise ValueError(f"Unsupported dataset: {args.data}")
    
    log_info(f"Using data_end={args.data_end} for dataset {args.data}")

    reward_model_architecture = None
    model_config_json = None
    try:
        with open(os.path.join(args.reward_model_path, "config.json"), "r") as f:
            model_config_json = json.load(f)
        reward_model_architecture = model_config_json["architectures"][0]
    except FileNotFoundError:
        raise FileNotFoundError(f"config.json not found in {args.reward_model_path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in config.json: {e}")
    except KeyError:
        raise KeyError("'architecture' key not found in config.json")

    try:
        check_chat_template_jinja(
            reward_model_path=args.reward_model_path,
            reward_model_architecture=reward_model_architecture,
            model_config_json=model_config_json,
        )
    except Exception as e:
        raise ValueError(
            f"Error when checking chat template for {reward_model_architecture}: {e}"
        )

    try:
        check_preprocessor_config(args.reward_model_path, reward_model_architecture)
    except Exception as e:
        raise ValueError(
            f"Error when checking preprocessor config for {reward_model_architecture}: {e}"
        )

    os.makedirs(args.output_dir, exist_ok=True)

    # Wrap main execution in try-catch for error notifications
    try:
        # Branch dataset loading based on dataset type
        # Dataset detection and routing
        if args.data == "mathvista_testmini":
            if "minicpm" in args.policy_model_path.lower():
                log_info("Loading MathVista dataset...")
                dataset_df = load_mathvista_dataset_from_file('/scratch_aisg/SPEC-SF-AISG/ob1/mmr-eval/evaluation/base_model_eval_code/data', 'MathVista_MINI')
            else:
                log_info("Loading MathVista dataset...")
                dataset_df = load_mathvista_dataset('AI4Math/MathVista', 'testmini')
            dataset_info = {
                "name": args.data,
                "answer_key": "answer",
                "interleave_image_tokens": False,
                "is_puzzle": False
            }
        elif args.data == "MMMU_DEV_VAL":
            # Load MMMU dataset using existing approach
            dataset_df = load_mmmu_dataset(args.data_dir, args.data)
            dataset_info = {
                "name": args.data,
                "answer_key": "answer", 
                "interleave_image_tokens": True,
                "is_puzzle": False
            }
        elif args.data == "puzzleVQA_1K_subset":
            dataset_df = load_puzzle_subset(str(PUZZLE_VQA_DATA), "puzzleVQA_1K_subset")
            dataset_info = get_puzzle_dataset_info("puzzleVQA_1K_subset")
            dataset_info["is_puzzle"] = True
            dataset_info["is_1k_subset"] = True
            dataset_info["interleave_image_tokens"] = False
        elif args.data == "AlgoPuzzleVQA_900_subset":
            dataset_df = load_puzzle_subset(str(ALGO_PUZZLE_VQA_DATA), "AlgoPuzzleVQA_900_subset")
            dataset_info = get_puzzle_dataset_info("AlgoPuzzleVQA_900_subset")
            dataset_info["is_puzzle"] = True
            dataset_info["is_1k_subset"] = True
            dataset_info["interleave_image_tokens"] = False
        elif args.data == "mathvision_test":
            log_info("Loading MathVision dataset...")
            dataset_df = load_mathvision_dataset('MathLLMs/MathVision', 'test')
            dataset_info = {
                "name": args.data,
                "answer_key": "answer",
                "interleave_image_tokens": False,
                "is_puzzle": False
            }
        else:
            raise ValueError(f"Unsupported dataset: {args.data}")

        # Extract dataset configuration
        ans_key = dataset_info["answer_key"]
        interleave_image_tokens = dataset_info["interleave_image_tokens"]

        log_info(f"Loaded Policy Model: {args.policy_model_path}")
        log_info(f"Loaded Reward Model: {args.reward_model_path}")

        dataset = dataset_df

        # Use provided run_datetime if available (for parallel runs), otherwise generate new one
        if args.run_datetime:
            run_datetime = args.run_datetime
        else:
            run_datetime = datetime.datetime.now().strftime(
                "%Y%m%d_%H%M%S"
            )  # for filename below
        log_info(f"Result file suffix: {run_datetime}")
        log_info(f"Evaluating dataset: {args.data}")
        
        # Dataset is a pandas DataFrame
        log_info(f"Dataset columns: {list(dataset.columns)}")

        if args.development_mode:
            log_info("Development mode is True, filtering dataset for testing")
            if dataset_info["is_puzzle"]:
                if dataset_info.get("is_1k_subset", False):
                    # For PuzzleVQA 1K subset: use first 2 samples from each puzzle type (40 total)
                    log_info(f"{dataset_info['name']} development mode: using first 2 samples per puzzle type")
                    dev_samples = []
                    test_puzzles = PUZZLE_DATASET_TYPES if dataset_info["name"] == "puzzleVQA_1K_subset" else ALGOPUZZLE_DATASET_TYPES
                    for puzzle_type in test_puzzles:
                        puzzle_samples = dataset[dataset["puzzle_type"] == puzzle_type].head(1)
                        dev_samples.append(puzzle_samples)
                    dataset = pd.concat(dev_samples, ignore_index=True) if dev_samples else dataset.iloc[:40]
                else:
                    # For individual PuzzleVQA or AlgoPuzzleVQA: use first 8 samples
                    log_info(f"{dataset_info['name']} development mode: using first 8 samples")
                    dataset = dataset.iloc[:8].reset_index(drop=True)
            elif dataset_info["name"] == "mathvista_testmini":
                # Use first 8 samples for MathVista development testing
                target_ids = ["1", "2", "3", "4", "5", "6", "7", "8"]
                if "minicpm" in args.policy_model_path.lower():
                    target_ids = [1, 2, 3, 4, 5, 6, 7, 8]
                    dataset = dataset[dataset["index"].isin(target_ids)]
                else:
                    dataset = dataset[dataset["id"].isin(target_ids)]
                log_info(f"MathVista development mode: Using target_ids: {target_ids}")

            elif dataset_info["name"] == "mathvision_test":
                # Use first 8 samples for MathVista development testing
                target_ids = ["1", "2", "3", "4", "5", "6", "27", "39"]
                dataset = dataset[dataset["id"].isin(target_ids)]
                log_info(f"MathVision development mode: Using target_ids: {target_ids}")

            elif dataset_info["name"] == "MMMU_DEV_VAL":  # MMMU
                    target_ids = [
                        "validation_Math_19",
                        "validation_Biology_24", 
                        "validation_Biology_29",
                        "validation_Accounting_1",
                        "validation_Accounting_2",
                        "validation_Accounting_3",
                        "validation_Accounting_4",
                        "validation_Accounting_5",
                    ]
                    dataset = dataset[dataset["id"].isin(target_ids)]
                    log_info("MMMU development mode: Using target_ids")
        else:
            log_info("Development mode is False, evaluating on full dataset")

        if not isinstance(args.data_begin, int) or not isinstance(
            args.data_end, int
        ):
            raise ValueError("data_begin and data_end must be integers")
        if args.data_end <= args.data_begin:
            raise ValueError(
                f"data_end ({args.data_end}) must be greater than data_begin ({args.data_begin})"
            )
        if args.data_begin < 0:
            raise ValueError(f"data_begin ({args.data_begin}) must be non-negative")

        start_idx = args.data_begin
        end_idx = min(args.data_end, len(dataset))
        
        data = dataset.iloc[start_idx:end_idx].reset_index(drop=True)
            
        log_info(
            f"Using range [{args.data_begin}, {args.data_end}): Data size after selection: {len(data)}"
        )

        # Ensure CUDA device is available before initializing, sometimes fails because of failed CUDA
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        
        available_gpus = torch.cuda.device_count()
        log_info(f"Available GPUs: {available_gpus}")
        log_info(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
        
        if args.policy_gpu >= available_gpus:
            raise ValueError(f"Policy GPU {args.policy_gpu} not available. Only {available_gpus} GPUs found.")
        if args.reward_gpu >= available_gpus:
            raise ValueError(f"Reward GPU {args.reward_gpu} not available. Only {available_gpus} GPUs found.")
        
        log_info(f"Initializing policy model on GPU index {args.policy_gpu} (cuda:{args.policy_gpu})")
        log_info(f"Initializing reward model on GPU index {args.reward_gpu} (cuda:{args.reward_gpu})")
        
        # VLLM engine configurations for different model sizes
        if "13b" in args.policy_model_path.lower(): # LlavaNext 13B
            policy_model = LLM(
                model=args.policy_model_path,
                max_num_seqs=64,
                gpu_memory_utilization=0.75,
                max_model_len=4096,  # Adjust based on your GPU memory
                limit_mm_per_prompt={"image": 1}  # Limit to 1 image per prompt
            )
        elif args.policy_model_path == "openbmb/MiniCPM-V-2_6":
            policy_model = LLM(
                model=args.policy_model_path,
                max_num_seqs=128,
                limit_mm_per_prompt={"image": 24},
                gpu_memory_utilization=0.80,
                trust_remote_code=True,
            )
        elif "7b" in args.policy_model_path.lower():
            policy_model = LLM(
                model=args.policy_model_path,
                max_num_seqs=128,
                limit_mm_per_prompt={"image": 24},
                gpu_memory_utilization=0.80,
                mm_processor_kwargs={
                    "min_pixels": 256 * 28 * 28,
                    "max_pixels": 1280 * 28 * 28,
                },
            )
        elif "3b" in args.policy_model_path.lower():
            policy_model = LLM(
                model=args.policy_model_path,
                max_num_seqs=128,
                limit_mm_per_prompt={"image": 24},
                gpu_memory_utilization=0.80,
                mm_processor_kwargs={
                    "min_pixels": 256 * 28 * 28,
                    "max_pixels": 1280 * 28 * 28,
                },
            )
        elif "32b" in args.policy_model_path.lower():
            policy_model = LLM(
                model=args.policy_model_path,
                max_num_seqs=32,
                limit_mm_per_prompt={"image": 24},
                gpu_memory_utilization=0.90,
                mm_processor_kwargs={
                    "min_pixels": 256 * 28 * 28,
                    "max_pixels": 1280 * 28 * 28,
                },
            )
        elif "72b" in args.policy_model_path.lower():
            policy_model = LLM(
                model=args.policy_model_path,
                max_num_seqs=16,
                limit_mm_per_prompt={"image": 24},
                gpu_memory_utilization=0.95,
                mm_processor_kwargs={
                    "min_pixels": 256 * 28 * 28,
                    "max_pixels": 1280 * 28 * 28,
                },
            )
        elif "gemma-3-12b-it" in args.policy_model_path.lower():
            policy_model = LLM(
                model=args.policy_model_path,
                max_num_seqs=64,
                gpu_memory_utilization=0.75,
                limit_mm_per_prompt={"image": 24},
                mm_processor_kwargs={
                    "do_pan_and_scan": True,
                },
            )
        elif "gemma-3-27b-it" in args.policy_model_path.lower():
            policy_model = LLM(
                model=args.policy_model_path,
                max_num_seqs=32,
                gpu_memory_utilization=0.88,
                limit_mm_per_prompt={"image": 24},
                mm_processor_kwargs={
                    "do_pan_and_scan": True,
                },
            )
        else:
            raise ValueError(f"Unsupported policy model: {args.policy_model_path}")

        if "llava" in args.policy_model_path.lower():
            policy_model_processor = LlavaNextProcessor.from_pretrained(args.policy_model_path, patch_size=14) # hardcode for now
        elif "qwen" in args.policy_model_path.lower():
            policy_model_processor = AutoProcessor.from_pretrained(args.policy_model_path, min_pixels=256 * 28 * 28, max_pixels=1280 * 28 * 28) # config from Qwen2.5-VL-7B-Instruct
        elif "gemma" in args.policy_model_path.lower():
            # policy_model_processor = AutoProcessor.from_pretrained(args.policy_model_path)
            policy_model_processor = Gemma3Processor.from_pretrained(
                args.policy_model_path,
                do_pan_and_scan=True
            )
        elif "minicpm" in args.policy_model_path.lower():
            policy_model_processor = AutoTokenizer.from_pretrained(args.policy_model_path, trust_remote_code=True)
            torch.cuda.empty_cache()
        else:
            raise ValueError(f"Unsupported policy model: {args.policy_model_path}")

        # Override for Gemma models due to incorrect tokenizer tokens
        if "gemma" in args.policy_model_path.lower():
            policy_model_processor.tokenizer.eos_token = "<end_of_turn>"
            policy_model_processor.tokenizer.bos_token = "[BOS]"
            log_info(
                f"Overriding EOS token for Gemma model {args.policy_model_path}: {policy_model_processor.tokenizer.eos_token}"
            )
            log_info(
                f"Overriding BOS token for Gemma model {args.policy_model_path}: {policy_model_processor.tokenizer.bos_token}"
            )
        elif "cpm" in args.policy_model_path.lower() or "internvl" in args.policy_model_path.lower(): 
            log_info(
                f"Using EOS token for policy model {args.policy_model_path}: {policy_model_processor.eos_token}"
            )
            log_info(
                f"Using BOS token for policy model {args.policy_model_path}: {policy_model_processor.bos_token}"
            )
        else:
            log_info(
                f"Using EOS token for policy model {args.policy_model_path}: {policy_model_processor.tokenizer.eos_token}"
            )
            log_info(
                f"Using BOS token for policy model {args.policy_model_path}: {policy_model_processor.tokenizer.bos_token}"
            )

        if "cpm" in args.policy_model_path.lower() or "internvl" in args.policy_model_path.lower():
            EOS_TOKEN = policy_model_processor.eos_token
        else:
            EOS_TOKEN = policy_model_processor.tokenizer.eos_token

        reward_model_init_kwargs = {
            "torch_dtype": torch.bfloat16,
            "attn_implementation": "flash_attention_2",
            "device_map": f"cuda:{args.reward_gpu}",
        }

        reward_model = VisualPRM(args.reward_model_path, reward_model_init_kwargs)

        new_dataset = []
        
        # Iterate through pandas DataFrame
        for i in tqdm(range(len(data)), desc="Running inference"):
            line = data.iloc[i]
            # dump_image(line, img_root) # for when first time running and need to convert b64 images to local images
            # index = line["index"]
            line_dict = line.to_dict()
            for k, v in line_dict.items():
                if isinstance(v, np.integer):
                    line_dict[k] = int(v)
                elif isinstance(v, np.floating):
                    line_dict[k] = float(v)

            # Extract image data - handle both MMMU and PuzzleVQA formats
            image_data_base64_list = []  # just a list of b64 strings, not objects
            
            if dataset_info["is_puzzle"]:
                # PuzzleVQA format: single base64 image in "image" field, because we already wrote the processing in load_puzzlevqa_1k_subset function (original PuzzleVQA JSONL file is a path to local image)
                if "image" in line_dict and line_dict["image"]:
                    image_data_base64_list.append(line_dict["image"])
            else:
                # MMMU format: potentially multiple images in "image" field
                if "image" in line_dict and line_dict["image"]:
                    # Handle single image case
                    if isinstance(line_dict["image"], str):
                        image_data_base64_list.append(line_dict["image"])
                    # Handle multiple images case
                    elif isinstance(line_dict["image"], list):
                        image_data_base64_list.extend(
                            [img for img in line_dict["image"] if img]
                        )
            
            log_info(f"Found {len(image_data_base64_list)} base64 images in data")
            if len(image_data_base64_list) == 0:
                raise ValueError(f"No image data found for data sample: {line_dict}")

            question = line["question"]
            if "Answer Yes or No." in question:
                question = question.replace("Answer Yes or No.", "")
            
            detailed_cot_prompt = "If you are uncertain or the problem is too complex, make a reasoned guess based on the information provided. Avoid repeating steps indefinitely—provide your best guess even if unsure. Determine whether to think step by step based on the difficulty of the question, considering all relevant information before answering. If you do need to think step by step, first, carefully observe and describe everything you see: the image itself and how it connects to the problem being presented. Then, work through your reasoning step by step to arrive at your answer. **Your teacher will review your descriptions of visual elements to ensure you're observing all relevant details accurately, and will critique each step of your reasoning to provide guidance and ensure you're on the right track.** Put your final answer within \\boxed{}. If multiple-choice options for answers to the question are provided, select the alphabetical letter of the corresponding best option within \\boxed{} when you are ready to provide your final answer."

            minicpm_cot_prompt = ('''Read the question carefully and analyze all problem-relevant visual elements through a step-by-step perceptual process. Determine whether to think step by step based on the difficulty of the question, considering all relevant information before answering. If you do need to think step by step, describe what you see step by step, solve it step by step, and '''
                                     '''then output the final answer in the format of "Answer: single number '''
                                     '''or single word or phrase".\n\n''')
            minicpm_cot_prompt += "When working out your reasoning step by step, 请一步步推理, especially when that helps clarity and accuracy."

            if dataset_info["name"] == "mathvista_testmini":
                if "minicpm" in args.policy_model_path.lower():
                    question = line_dict["question"]
                    user_prompt = build_minicpm_mathvista_cot_prompt(line_dict)
                else:
                    question = line_dict["question"]
                    user_prompt = build_mathvista_prompt(line_dict)

            elif dataset_info["name"] == "mathvision_test":
                question = line_dict["question"]
                user_prompt = build_mathvision_prompt(line_dict)
                if "minicpm" in args.policy_model_path.lower():
                    user_prompt += minicpm_cot_prompt
                else:
                    user_prompt += detailed_cot_prompt

            elif dataset_info["is_puzzle"]:
                options_list = line_dict["options"]
                # user_prompt = build_puzzlevqa_prompt(question, options_list)
                if dataset_info["name"] == "puzzleVQA_1K_subset":
                    user_prompt = build_puzzlevqa_prompt(question, options_list)
                    if "minicpm" in args.policy_model_path.lower():
                        user_prompt = build_puzzlevqa_prompt_minicpm(question, options_list)
                elif dataset_info["name"] == "AlgoPuzzleVQA_900_subset":
                    user_prompt = build_algopuzzlevqa_prompt(question, options_list)
                    if "minicpm" in args.policy_model_path.lower():
                        user_prompt = build_algopuzzlevqa_prompt_minicpm(question, options_list)
            else:
                # MMMU prompt format
                options = {
                    cand: line_dict[cand]
                    for cand in string.ascii_uppercase
                    if cand in line_dict and not pd.isna(line_dict[cand])
                }
                user_prompt = ""
                
                # Add hint if present
                if "hint" in line_dict and not pd.isna(line_dict["hint"]):
                    user_prompt += f"Hint: {line_dict['hint']}\n"
                
                user_prompt += f"Question: {question}\n"

                if len(options):
                    user_prompt += "Options:\n"
                    for key, item in options.items():
                        user_prompt += f"{key}. {item}\n"
                
                user_prompt += "Please select the correct answer from the options above.\n"
                if "minicpm" in args.policy_model_path.lower():
                    user_prompt += minicpm_cot_prompt
                else:
                    user_prompt += detailed_cot_prompt
                # user_prompt += "If you are uncertain or the problem is too complex, make a reasoned guess based on the information provided. Avoid repeating steps indefinitely—provide your best guess even if unsure. Determine whether to think step by step based on the difficulty of the question, considering all relevant information before answering. If you do need to think step by step, first, carefully observe and describe everything you see: the image itself and how it connects to the problem being presented. Then, work through your reasoning step by step to arrive at your answer. **Your teacher will review your descriptions of visual elements to ensure you're observing all relevant details accurately, and will critique each step of your reasoning to provide guidance and ensure you're on the right track.** Put your final answer within \\boxed{}. If multiple-choice options for answers to the question are provided, select the alphabetical letter of the corresponding best option within \\boxed{} when you are ready to provide your final answer."

                user_prompt = user_prompt.rstrip()

            log_info(f"Built prompt before interleave_image_tokens: {user_prompt}")

            # TODO: For now we use the same function for both MMMU and PuzzleVQA, but be careful when using for MathVista
            if "internvl" in args.policy_model_path.lower() or "minicpm" in args.policy_model_path.lower():
                question_in_messages_array_format, question_corresponding_image_data_base64_list = (
                    prepare_question_array_with_base64_image_strings_for_internvl_and_minicpm(
                        user_prompt,
                        image_data_base64_list,
                        interleave_image_tokens=interleave_image_tokens,
                        model_type="internvl" if "internvl" in args.policy_model_path.lower() else "minicpm"
                    )
                )
            elif dataset_info["name"] == "mathvision_test":
                question_in_messages_array_format, question_corresponding_image_data_base64_list = (
                    prepare_question_array_with_base64_image_strings_mathvision(
                        user_prompt,
                        image_data_base64_list,
                        interleave_image_tokens=interleave_image_tokens,
                    )
                )
            else:
                question_in_messages_array_format, question_corresponding_image_data_base64_list = (
                    prepare_question_array_with_base64_image_strings(
                        user_prompt,
                        image_data_base64_list,
                        interleave_image_tokens=interleave_image_tokens,
                    )
                )
            log_info(f"length of question_corresponding_image_data_base64_list: {len(question_corresponding_image_data_base64_list)}")
            
            # Use simple system prompt like vllm_run_mmmu.py
            # system_prompt = "You are a helpful assistant."
            system_prompt = POLICY_VISUAL_ANALYST_SYS_PROMPT_V3
            if "minicpm" in args.policy_model_path.lower():
                system_prompt += " When working through this problem, you may use Mandarin for your internal reasoning if it helps you think more clearly or access relevant concepts."

            sys_and_user_prompt_messages_array = (
                [
                    {"role": "system", "content": system_prompt},
                ]
                + question_in_messages_array_format
                # + [{"role": "assistant", "content": ""}]
            )

            user_input_prompt = policy_model_processor.apply_chat_template(
                sys_and_user_prompt_messages_array,
                tokenize=False,
                add_generation_prompt=True,
            )
            # # Replace only the last EOS_TOKEN occurrence
            # parts = user_input_prompt.rsplit(EOS_TOKEN, 1)
            # if len(parts) == 2:
            #     user_input_prompt = parts[0] + parts[1]
            user_input_prompt = user_input_prompt.strip()
            if "gemma" in args.policy_model_path.lower():
                user_input_prompt += "\n\n" # for Gemma models which have a weird "model" token that the model does not recognize is not part of the text
            log_info(f"Input prompt for generations: {user_input_prompt}")

            previous_steps = []
            # used_steps = set() # TODO: Consider using this later to avoid selecting the same steps already in trace, but requires more downstream adaptation as well. Currently if death spiral cos of repetition, likely cuts off due to max_iterations reached, returns None answer.
            max_iteration = 100  # TODO: set this accordingly to the max number of steps we allow model to generate
            iteration_history = []
            iteration_index = 0
            pred_answer = None

            while max_iteration > 0:
                log_info(
                    f"Iteration {iteration_index} and previous steps: {previous_steps}"
                )
                max_iteration -= 1
                iteration_index += 1

                new_steps_candidates = get_next_step(
                    policy_model,
                    question_corresponding_image_data_base64_list,
                    previous_steps,
                    user_input_prompt,
                    EOS_TOKEN,
                    policy_model_processor,
                    sys_and_user_prompt_messages_array,
                )  # Best of N starts here
                if len(new_steps_candidates) == 0:
                    log_info(
                        f"Question Number: {i}, No candidate generated at iteration {iteration_index}."
                    )
                    continue
                log_info(
                    f"Question Number: {i}, Iteration: {iteration_index}, New Steps Candidates Number: {len(new_steps_candidates)}"
                )

                iteration_data = {
                    "iteration_index": iteration_index,
                    "candidates_info": [],
                    "chosen_step_index": None,
                    "chosen_step": None,
                }
                
                for candidate_step in new_steps_candidates:
                    # if candidate_step["text"] in used_steps:
                    #     log_info(
                    #         f"Skipping already used candidate: {candidate_step['text'][:100]}..."
                    #     )
                    #     continue
                    # used_steps.add(candidate_step["text"])
                    
                    # Check if candidate_step ends with EOS_TOKEN to denote end of generating, answer/final conclusion is reached by model
                    if candidate_step["text"].endswith(EOS_TOKEN):
                        temp_candidate_step = candidate_step["text"].replace(EOS_TOKEN, "")
                        log_info("Found step with EOS_TOKEN, removed for processing")
                    else:
                        temp_candidate_step = candidate_step["text"]

                    log_info(
                        f"Getting reward for temp candidate step: {temp_candidate_step}"
                    )
                    reward_result = reward_model.get_reward(
                        user_prompt,  # question including MCQ options if relevant
                        previous_steps,
                        temp_candidate_step,
                        question_corresponding_image_data_base64_list, # use the same processed images as policy model
                        interleave_image_tokens=interleave_image_tokens,
                    )
                    log_info(f"Reward result returned: {reward_result}")

                    reward_score = reward_result["reward_score"]
                    if reward_score == -1:
                        log_info(
                            f"Reward score is -1 for candidate step: {temp_candidate_step}"
                        )
                        # Don't skip anymore - we'll store the negative probability for potential selection
                        reward = reward_score
                    else:
                        reward = reward_score

                    # log_info(
                    #     f"Question Number: {i}, Previous Steps Number: {len(previous_steps)}, Candidate Step Index: {candidate_idx}, Reward: {reward}"
                    # )

                    candidate_info = {
                        "candidate_step": candidate_step["text"],
                        "reward": reward,
                        "reward_result": reward_result,  # Store full reward information including probabilities
                    }
                    iteration_data["candidates_info"].append(candidate_info)

                # Check if any candidate had a boxed answer (early termination flag)
                # if early_termination:
                #     log_info(
                #         f"Question Number: {i}, Found boxed answer at iteration {iteration_index}, will break after step selection."
                #     )

                # Enhanced Greedy Selection:
                # 1. First try to select candidate with highest positive reward
                # 2. If all rewards are -1, select candidate where the model is least confident that the step is a bad step,
                # this way we do not cut off midway, and we can still get to a final answer
                max_reward = -1
                max_reward_idx = -1

                # First pass: look for positive rewards
                for idx, candidate_info in enumerate(iteration_data["candidates_info"]):
                    if candidate_info["reward"] > max_reward:
                        max_reward = candidate_info["reward"]
                        max_reward_idx = idx

                # If no positive rewards found, select based on least negative score
                if max_reward_idx == -1:
                    log_info(
                        "All candidates have negative predictions, selecting based on least negative probability"
                    )
                    best_negative_prob = float("inf")  # Initialize to high value
                    best_negative_idx = -1

                    for idx, candidate_info in enumerate(
                        iteration_data["candidates_info"]
                    ):
                        if (
                            candidate_info["reward"] == -1
                        ):  # Only consider negative predictions
                            negative_prob = candidate_info["reward_result"][
                                "negative_prob"
                            ]
                            log_info(
                                f"Candidate {idx} negative probability: {negative_prob}"
                            )

                            # Select candidate with lowest negative probability (least negative)
                            if negative_prob < best_negative_prob:
                                best_negative_prob = negative_prob
                                best_negative_idx = idx

                    if best_negative_idx != -1:
                        max_reward_idx = best_negative_idx
                        log_info(
                            f"Selected candidate {best_negative_idx} with smallest probability (least confident in being an incorrect step) {best_negative_prob}"
                        )
                    else:
                        log_info(
                            "Failed to select the least negative probability candidate somehow, using first available candidate as fallback"
                        )  # happens when SINGLE false negative <|im_end|> token is output by model, and there are no other candidate steps at this step
                        if len(iteration_data["candidates_info"]) > 0:
                            max_reward_idx = 0  # Select first candidate as fallback
                            log_info(
                                f"Selected first candidate as fallback: {iteration_data['candidates_info'][0]['candidate_step']}"
                            )
                        else:
                            log_info(
                                "No candidates available at all, breaking from iteration loop"
                            )
                            break

                # Check if we have a valid selection before proceeding
                if max_reward_idx == -1:
                    log_info(
                        f"Question Number: {i}, No valid candidate selected, breaking from iteration loop"
                    )
                    break

                iteration_data["chosen_step_index"] = max_reward_idx
                iteration_data["chosen_step"] = iteration_data["candidates_info"][
                    max_reward_idx
                ]["candidate_step"]
                log_info(
                    f"Question Number: {i}, Chosen Step: {iteration_data['chosen_step']}"
                )
                # log_info(
                #     f"Question Number: {i}, appending step to previous steps: {iteration_data['chosen_step']}"
                # )
                # if iteration_data["chosen_step"] not in previous_steps:
                previous_steps.append(iteration_data["chosen_step"])
                iteration_history.append(iteration_data)

                # checking if the chosen_step has reached the end of the answer
                if len(previous_steps) > 0 and (
                    EOS_TOKEN in previous_steps[-1]
                ):
                    log_info(
                        f"Question Number: {i}, Early stopping at iteration {iteration_index}."
                    )
                    pred_answer = None

                    pred_answer = extract_boxed(previous_steps[-1])
                    log_info(f"extract_boxed extracted answer: {pred_answer} from final step")
                    
                    # If no boxed answer found, use the full text for later evaluation
                    if pred_answer is None:
                        pred_answer = previous_steps[-1]  # Store full text for eval stage
                        log_info("No boxed answer found, storing full text for evaluation")
                    
                    # Extract answer based on dataset type
                    # if dataset_info["is_puzzle"]:
                    #     # PuzzleVQA answer extraction
                    #     options_list = line_dict["options"]
                    #     pred_answer = extract_puzzlevqa_answer(previous_steps[-1], options_list)
                    #     log_info(f"PuzzleVQA extracted answer: {pred_answer} from final step")
                    # else:
                    #     # MMMU answer extraction (existing logic)
                    #     pred_answer = extract_boxed(previous_steps[-1])
                    #     log_info(f"MMMU extracted answer: {pred_answer} from final step")
                        
                    #     # If no boxed answer found, use the full text for later evaluation
                    #     if pred_answer is None:
                    #         pred_answer = previous_steps[-1]  # Store full text for eval stage
                    #         log_info("No boxed answer found, storing full text for evaluation")
                    break

            if "minicpm" in args.policy_model_path.lower():
                if len(previous_steps) == 1:
                    raw_full_prediction = previous_steps[0]
                elif len(previous_steps) > 1:
                    raw_full_prediction = "\n\n".join(previous_steps)
                else:
                    raise ValueError(f"No previous steps found: {previous_steps}")
                # log_info(f"line_dict: {line_dict}")
                # log_info(f"line: {line}")
                raw_full_prediction = raw_full_prediction.replace("<|im_end|>", "")

                gt_answer = None # quirk of using the .tsv file for loading MathVista is that the answer is not in the same format as the other datasets, and hence the need to check for question_type. 
                if dataset_info["name"] == "mathvista_testmini":
                    question_type = line_dict.get("question_type", "")

                    if question_type == "multi_choice":
                        gt_answer = line_dict.get("answer_option", "")
                    elif question_type == "free_form":
                        gt_answer = line_dict.get("answer", "")
                else:
                    gt_answer = line[ans_key]
                result_record = {
                    "question": question,
                    "index": int(line["index"])
                    if isinstance(line["index"], np.integer)
                    else line["index"],
                    "annotation": line_dict,
                    "task": args.data,
                    "iteration_history": iteration_history,
                    "gt_answer": gt_answer,
                    "raw_full_prediction": raw_full_prediction,
                    "pred_answer": None,  # as we are using non-CoT and want model to answer directly
                    # "pred_answer": extract_boxed(pred_answer),  # use this for LLM Judge
                }
            else:
                # Prepare result record
                result_record = {
                    "question": question,
                    "index": int(line["index"]) if isinstance(line["index"], np.integer) else line["index"],
                    "annotation": line_dict,
                    "task": args.data,
                    "iteration_history": iteration_history,
                    "final_steps": previous_steps,
                    "gt_answer": line[ans_key],
                    "pred_answer": pred_answer,
                }
            
            # Add puzzle_type for PuzzleVQA 1K subset analysis
            if dataset_info["is_puzzle"] and "puzzle_type" in line_dict:
                result_record["puzzle_type"] = line_dict["puzzle_type"]
            
            new_dataset.append(result_record)

        # Write results to output file after processing all data
        # Add partition suffix if running in parallel mode
        if args.partition_id is not None:
            output_file = os.path.join(
                args.output_dir,
                f"result-p{args.partition_id}-{args.data_begin}-{args.data_end}-{run_datetime}.json",
            )
        else:
            output_file = os.path.join(
                args.output_dir,
                f"result-{args.data_begin}-{args.data_end}-{run_datetime}.json",
            )
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(new_dataset, f, ensure_ascii=False, indent=2)

        log_info(f"Done! Results are saved to {output_file}.")

        # Calculate evaluation score based on dataset type
        log_info("Calculating evaluation score...")
        # # if dataset_info["is_puzzle"]:
        # #     if dataset_info.get("is_1k_subset", False):
        # #         # Use detailed PuzzleVQA 1K subset evaluation with per-puzzle-type breakdown
        # #         eval_score, correct_count, total_count, per_puzzle_scores = calculate_puzzlevqa_1k_subset_detailed_score(output_file)
        # #     else:
        # #         # Use standard PuzzleVQA evaluation for individual puzzle types
        # #         eval_score, correct_count, total_count = calculate_puzzlevqa_score(output_file)
        # else:
        #     # Use MMMU evaluation (existing logic)
        #     eval_score, correct_count, total_count = calculate_evaluation_score_direct(output_file)

        if "minicpm" in args.policy_model_path.lower():
            log_info("No direct eval score calculation for MiniCPM")
            eval_score, correct_count, total_count = None, None, None
        else:
            eval_score, correct_count, total_count = calculate_evaluation_score_direct(output_file)
        
        if eval_score is not None:
            eval_score_percent = f"{eval_score:.2%}"
            eval_summary = (
                f"Score: {eval_score_percent} ({correct_count}/{total_count} correct)"
            )
        else:
            eval_score_percent = "N/A"
            eval_summary = "Score calculation failed"

        log_info(f"Evaluation summary: {eval_summary}")


    except Exception as e:
        error_message = str(e)
        error_traceback = traceback.format_exc()
        log_info(f"Evaluation failed with error: {error_message}")
        log_info(f"Full traceback: {error_traceback}")
        
        # Check if this is an OOM error
        is_oom = any(oom_indicator in str(e).lower() or oom_indicator in error_traceback.lower() 
                     for oom_indicator in ['out of memory', 'oom', 'cuda out of memory', 'memory error'])

        # Send error notification via Telegram
        try:
            # Add OOM indicator and partition info to the message
            error_msg_prefix = ""
            if is_oom:
                error_msg_prefix = "🚨 OOM ERROR: "
            if args.partition_id is not None:
                error_msg_prefix += f"Partition {args.partition_id} failed - "
                
            # send_telegram_error_notification(
            #     model_path_name=args.policy_model_path,
            #     error_message=error_msg_prefix + error_message,
            #     error_traceback=error_traceback,
            #     evaluation_run_logs_file=os.getenv("EVAL_RUN_LOG_FILE", ""),
            #     extra_fields={
            #         "reward_model_path": args.reward_model_path,
            #         "data": args.data,
            #         "data_begin": args.data_begin,
            #         "data_end": args.data_end,
            #         "development_mode": args.development_mode,
            #         "partition_id": args.partition_id,
            #         "is_oom_error": is_oom,
            #         "policy_gpu": args.policy_gpu,
            #         "reward_gpu": args.reward_gpu,
            #     },
            #     send_files=True,
            # )
        except Exception as telegram_error:
            log_info(f"Failed to send error notification to Telegram: {telegram_error}")

        # Re-raise the original exception to maintain proper exit code
        raise


if __name__ == "__main__":
    main()