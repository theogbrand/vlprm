import os
import numpy as np
import json
import base64
import math
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, Qwen2_5_VLForConditionalGeneration, AutoProcessor
import torch
from PIL import Image
from io import BytesIO
from eval.tts_eval.utils.utils import log_info
import argparse
from eval.tts_eval.reward_guided_search.prompts import PRM_SYSTEM_PROMPT_NORMAL_TOK_V2
from qwen_vl_utils import process_vision_info
from typing import List
from eval.tts_eval.reward_guided_search.utils.utils import prepare_question_array_with_base64_image_strings

# VLLM imports
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams

# Import InternVL image processing
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

POSITIVE_TOKEN = "+"
NEGATIVE_TOKEN = "-"

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image_from_base64(image_base64_list, input_size=448, max_num=12): # ref: https://huggingface.co/OpenGVLab/VisualPRM-8B-v1_1/blob/main/modeling_internvl_chat.py (def generate_steps_with_soft_score()), see VisualPRM_internvl_functions.py and check_intern_cpm.py for more details
    if isinstance(image_base64_list, str):
        image_base64_list = [image_base64_list] # This works for string containing a single b64 image string and for a string containing a list of b64 image strings
    
    all_pixel_values = []
    num_patches_list = []
    
    for image_base64 in image_base64_list:
        # Decode base64 to bytes first
        image_bytes = base64.b64decode(image_base64)
        # Open image from bytes
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        transform = build_transform(input_size=input_size)
        images = dynamic_preprocess(
            image, image_size=input_size, use_thumbnail=True, max_num=max_num
        )
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)  # Shape: (num_patches, 3, H, W)
        all_pixel_values.append(pixel_values)
        num_patches_list.append(pixel_values.shape[0])  # Track patches per image
    
    # Concatenate all images along batch dimension
    final_pixel_values = torch.cat(all_pixel_values, dim=0)
    return final_pixel_values, num_patches_list

# from prompts import POLICY_USER_PROMPT, MCQ_SUFFIX_PROMPT_NO_TEMPLATE
POLICY_USER_PROMPT = r"""Here is the question you need to answer: {{QUESTION}}"""
MCQ_SUFFIX_PROMPT_NO_TEMPLATE = """

Multiple-choice options:
{{OPTIONS}}
"""

def safe_isnan(val):
    try:
        return math.isnan(val)
    except (TypeError, ValueError):
        return False

def get_token_prob(logprobs_dict, target_token):
    """
    logprobs_dict: dict[token_id -> Logprob]
    target_token: string (decoded token to look for)
    """
    for tok_id, logprob_obj in logprobs_dict.items():
        if logprob_obj.decoded_token == target_token:
            return float(np.exp(logprob_obj.logprob))
    return 0  # token not found in top-k

PRM_SYSTEM_PROMPT_NORMAL_TOK = """**You are a process supervision model for visual reasoning tasks. You will receive an image and an image-based problem statement, followed by solution steps to evaluate.**

First round: problem statement and first solution step.  
Subsequent rounds: one new step per round.

Assess the cumulative correctness of the entire solution up to each step.

## Evaluation Criteria:

1. **Visual Accuracy**: Are visual elements from the image correctly identified (shapes, colors, positions, quantities, spatial relationships)?

2. **Logical Validity**: Do all inferences and calculations follow correctly from the image and previous steps?

## Response:
- **"+"** if correct up to this step
- **"-"** if any error exists up to this step

Only respond with "+" or "-". No explanations.

An error in any step invalidates all subsequent steps."""

def load_json(file_path):
    """
    Load a JSON file and return its contents as a Python object (dict or list).
    """
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def encode_image_to_base64(image_path):
    """Encode image to base64 string."""
    with Image.open(image_path) as img:
        # Convert RGBA to RGB if necessary
        if img.mode == "RGBA":
            img = img.convert("RGB")

        # Save to bytes buffer
        buffer = BytesIO()
        img.save(buffer, format="JPEG")
        buffer.seek(0)

        # Encode to base64
        return base64.b64encode(buffer.read()).decode("utf-8")

def save_json(data, filepath, indent=2):
    """
    Save Python object as JSON file, creating parent directories if needed.

    Args:
        data: Python object (dict, list, etc.)
        filepath: path to save the JSON file
        indent: indentation level for pretty printing (default=2)
    """
    # Ensure parent directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)

class InternVLVisualPRM:
    def __init__(self, args):
        self.pos_token = "+"
        self.neg_token = "-"
        self.wanted_tokens = ["+", "-"]

        # Set up sampling parameters
        self.sampling_params = SamplingParams(
            temperature=0,
            max_tokens=1,
            stop=None,
            guided_decoding=GuidedDecodingParams(choice = self.wanted_tokens),
            logprobs=10
        )

        self.tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, use_fast=False)
        self.PLACEHOLDER_TOKEN='+'
        # self.processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
        # existing_special_tokens = self.tokenizer.special_tokens_map.get("additional_special_tokens", [])
        # self.PLACEHOLDER_TOKEN='<+>'
        # new_special_tokens = [self.PLACEHOLDER_TOKEN]
        # updated_special_tokens = list(set(existing_special_tokens + new_special_tokens))
        # self.tokenizer.add_special_tokens({"additional_special_tokens": updated_special_tokens})
        # print(f"updated_special_tokens: {updated_special_tokens}")
        self.model = AutoModel.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
    ).eval().cuda()
        

    def get_agg_reward(self, image_strs, question, steps):
        pixel_values, num_patches_list = load_image_from_base64(image_strs)
        pixel_values = pixel_values.to(torch.bfloat16).to(self.model.device)

        if pixel_values is not None and '<image>' not in question:
            num_images = 1 if num_patches_list is None else len(num_patches_list)
            question = '<image>\n' * num_images + question

        # InternVL constants
        IMG_START_TOKEN = '<img>'
        IMG_END_TOKEN = '</img>'
        IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'
        PLACEHOLDER = self.PLACEHOLDER_TOKEN

        assert pixel_values is None or (len(pixel_values) == sum(num_patches_list) and len(num_patches_list) == question.count('<image>')), f'Error check: {len(pixel_values)=}, {sum(num_patches_list)=}, {len(num_patches_list)}, {question=}'
        
        # Set img_context_token_id for model (critical for InternVL)
        img_context_token_id = self.tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.model.img_context_token_id = img_context_token_id

        # Determine image input flag
        image_input = True # our use case always has image input
        
        # Get token IDs for "+" and "-" (following InternVL pattern)
        str2score = {'+': 1, '-': 0}
        candidate_tokens = []
        candidate_weights = []
        
        for k, v in str2score.items():
            k_id = self.tokenizer.convert_tokens_to_ids(k)
            assert k_id != self.tokenizer.unk_token_id
            candidate_tokens.append(k_id)
            candidate_weights.append(v)

        # Recreate conversation into single prompt with placeholder tokens to predict
        conversation_content = []
        for step_idx, step in enumerate(steps):
            if step_idx == 0:
                step_content = f'### Question:\n{question}\n\n### Solution Process:\n{step}'
            else:
                step_content = step
            conversation_content.append((step_content, PLACEHOLDER))
        
        # Build query in InternVL conversation format
        # Simplified template building (avoiding import issues)
        query_parts = [f"<|im_start|>system\n{PRM_SYSTEM_PROMPT_NORMAL_TOK}<|im_end|>"]
        
        for step_content, placeholder in conversation_content:
            query_parts.append(f"<|im_start|>user\n{step_content}<|im_end|>")
            query_parts.append(f"<|im_start|>assistant\n{placeholder}<|im_end|>")
        
        query = "\n".join(query_parts)

        # Calculate num_image_token like InternVL does: (image_size // patch_size)^2 * downsample_ratio^2  
        num_image_token = 2  # based on default values found in VisualPRM config
        for num_patches in num_patches_list:
            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * num_image_token * num_patches + IMG_END_TOKEN
            query = query.replace('<image>', image_tokens, 1)
        
        model_inputs = self.tokenizer(query, return_tensors='pt')
        input_ids = model_inputs['input_ids'].to(self.model.device)
        attention_mask = model_inputs['attention_mask'].to(self.model.device)
        image_flags = torch.tensor([image_input] * pixel_values.shape[0], dtype=torch.long).to(self.model.device)
        
        # Find placeholder positions - only those in exact "<|im_start|>assistant\n+<|im_end|>" pattern
        placeholder_positions = []
        input_ids_list = input_ids[0].tolist()
        placeholder_token_id = self.tokenizer.convert_tokens_to_ids(PLACEHOLDER)
        
        # Tokenize the complete pattern components
        im_start_assistant_tokens = self.tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
        im_end_tokens = self.tokenizer.encode("<|im_end|>", add_special_tokens=False)
        
        for i, token_id in enumerate(input_ids_list):
            if token_id == placeholder_token_id:
                # Check if this "+" is preceded by "<|im_start|>assistant\n" and followed by "<|im_end|>"
                has_prefix = False
                has_suffix = False
                
                # Check preceding tokens
                if i >= len(im_start_assistant_tokens):
                    preceding_tokens = input_ids_list[i - len(im_start_assistant_tokens):i]
                    if preceding_tokens == im_start_assistant_tokens:
                        has_prefix = True
                
                # Check following tokens
                if i + len(im_end_tokens) < len(input_ids_list):
                    following_tokens = input_ids_list[i + 1:i + 1 + len(im_end_tokens)]
                    if following_tokens == im_end_tokens:
                        has_suffix = True
                
                # Only add if both prefix and suffix match
                if has_prefix and has_suffix:
                    placeholder_positions.append(i)
        
        with torch.no_grad():
            # Forward pass following exact InternVL pattern
            logits = self.model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                image_flags=image_flags,
            ).logits

            logits = logits[0][placeholder_positions, :][:, candidate_tokens] # (+ token score, - token score) 
            # print(logits)
            soft_scores = logits.softmax(dim=-1).tolist()
            assert len(soft_scores) == len(steps), f"Error check: {len(soft_scores)=}, {len(steps)=}, {query=}"
            # Gather step scores
            steps_with_score = []
            total_score = 0
            for soft_score, step in zip(soft_scores, steps):
                score = 0
                for s, w in zip(soft_score, candidate_weights):
                    score += s * w
                total_score += score
                steps_with_score.append({'step': step, 'score': score})
            print(f"steps_with_score: {steps_with_score}")
            print(f"Average score: {total_score / len(steps)}")

            return total_score / len(steps)

class QwenVLVisualPRM:
    def __init__(self, model_path, model_init_kwargs=None):
        log_info(f"Loading model from {model_path}")

        if model_init_kwargs is None:
            model_init_kwargs = {
                "torch_dtype": torch.bfloat16,
                "attn_implementation": "flash_attention_2",
                "device_map": "auto",
            }
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path, **model_init_kwargs
        )
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path
        )
        self.processor = AutoProcessor.from_pretrained(model_path, min_pixels=256 * 28 * 28, max_pixels=1280 * 28 * 28)
        log_info("VisualPRM loaded successfully")
        self.pos_token_id = self.tokenizer.encode(POSITIVE_TOKEN)[0]
        self.neg_token_id = self.tokenizer.encode(NEGATIVE_TOKEN)[0]
        self.system_prompt = PRM_SYSTEM_PROMPT_NORMAL_TOK_V2 

    def inference_single(
        self, sample_input_messages_array_including_images_interweaved, logging=False
    ):
        # log_info(
        #     f"Starting inference with input: {sample_input_messages_array_including_images_interweaved}"
        # )

        text = self.processor.apply_chat_template(
            sample_input_messages_array_including_images_interweaved,
            tokenize=False,
            add_generation_prompt=True,
        )
        log_info(f"Reward model Text: {text}")

        # log_info("*" * 100)
        # log_info(sample_input_messages_array_including_images_interweaved)
        # log_info("*" * 100)
        # exit()
        
        image_inputs, video_inputs = process_vision_info(
            sample_input_messages_array_including_images_interweaved
        )

        log_info(f"CHECK: Len of Image inputs output from process_vision_info (should match len of base64 image list): {len(image_inputs)}")
        log_info(
            f"DEBUG: Image should be PIL Image as input to processor: {[type(img) if img else 'None' for img in image_inputs]}"
        )

        message_ids = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(self.model.device)        


        log_info(f"Tokenized message_ids shape: {message_ids['input_ids'].shape}")

        with torch.no_grad():
            outputs = self.model(
                **message_ids
            )  # [batch_size, seq_len, vocab_size] - double check size

        log_info(f"Model outputs logits shape: {outputs.logits.shape}")
        log_info(f"Model outputs logits device: {outputs.logits.device}")

        allowed_token_ids = torch.tensor([self.pos_token_id, self.neg_token_id], device=outputs.logits.device)  # shape: (2,)
        log_info(
            f"Allowed token IDs: {allowed_token_ids} (+ token: {self.pos_token_id}, - token: {self.neg_token_id})"
        )

        last_position_logits = outputs.logits[:, -1, :]  # [batch_size, vocab_size]
        log_info(f"Last position logits shape: {last_position_logits.shape}")
        log_info(
            f"Last position logits for + and - tokens: {last_position_logits[:, allowed_token_ids]}"
        )

        masked_logits = last_position_logits[:, allowed_token_ids]  # [batch_size, 2]
        log_info(f"Masked logits shape: {masked_logits.shape}")
        log_info(f"Masked logits values: {masked_logits}")

        probs_pos_neg = F.softmax(masked_logits, dim=-1)
        log_info(f"Probabilities [pos, neg]: {probs_pos_neg}")

        predicted_indices = masked_logits.argmax(dim=-1)
        predicted_tokens = allowed_token_ids[predicted_indices]
        log_info(f"Predicted indices: {predicted_indices}")
        log_info(f"Predicted token IDs: {predicted_tokens}")

        decoded_tokens = [self.tokenizer.decode([int(token_id)], skip_special_tokens=False) for token_id in predicted_tokens]
        log_info(f"Decoded predicted tokens: {decoded_tokens}")

        if logging:
            log_info(f"Decoded Labels (either + or -): {decoded_tokens}")

        positive_prob = probs_pos_neg[0][0].cpu().item()
        negative_prob = probs_pos_neg[0][1].cpu().item()
        
        if NEGATIVE_TOKEN in decoded_tokens:
            log_info("Negative prediction detected")
            reward_score = -1
        else:
            input_length = message_ids['input_ids'].shape[1]  # Total input tokens
            reward_score = (positive_prob ** 0.3) / (input_length ** 0.6)
            log_info(f"Normalized reward score: {reward_score}")

        result = {
            'prediction': 'negative' if NEGATIVE_TOKEN in decoded_tokens else 'positive',
            'positive_prob': positive_prob,
            'negative_prob': negative_prob,
            'reward_score': reward_score
        }
        
        log_info(f"Returning result: {result}")
        return result

    def get_reward(
        self,
        question: str,
        previous_steps: List[str],
        now_step: str,
        base64_image_list: List[str],
        interleave_image_tokens: bool = False,
    ) -> float:
        """
        Get reward score for a reasoning step given the question, previous steps, and current step.
        """

        messages_array_to_generate_reward = [
            # {"role": "system", "content": self.system_prompt}
            {
                "role": "system",
                "content": [{"type": "text", "text": self.system_prompt}],
            }
        ]  # mirrors training process


        if len(previous_steps) > 0:
            log_info(f"Previous steps > 0: {previous_steps}")
            for i, step in enumerate(previous_steps):
                if i == 0: # the first step requires to include question and set up solution process
                    standard_first_user_message = (
                        f"### Question:\n{question}\n\n### Solution Process:\n{step}"
                    )

                    standard_first_question_in_messages_array_format, standard_first_question_corresponding_image_data_base64_list = (
                        prepare_question_array_with_base64_image_strings(
                            standard_first_user_message,
                            base64_image_list,
                            interleave_image_tokens=interleave_image_tokens,
                        )
                    )

                    log_info(f"in VisualPRM length of standard_first_question_corresponding_image_data_base64_list: {len(standard_first_question_corresponding_image_data_base64_list)}")

                    messages_array_to_generate_reward += (
                        standard_first_question_in_messages_array_format
                    )
                    messages_array_to_generate_reward.append(
                        # {"role": "assistant", "content": POSITIVE_TOKEN}
                        {
                            "role": "assistant",
                            "content": [{"type": "text", "text": POSITIVE_TOKEN}],
                        }
                    )
                else: # subsequent steps after first step, we can just paste the previous step
                    messages_array_to_generate_reward.append(
                        # {"role": "user", "content": step}
                        {"role": "user", "content": [{"type": "text", "text": step}]}
                    ) 
                    messages_array_to_generate_reward.append(
                        # {"role": "assistant", "content": POSITIVE_TOKEN}
                        {
                            "role": "assistant",
                            "content": [{"type": "text", "text": POSITIVE_TOKEN}],
                        }
                    )
            
            messages_array_to_generate_reward.append(
                # {"role": "user", "content": now_step}
                {"role": "user", "content": [{"type": "text", "text": now_step}]}
            ) # set up for reward model to generate reward for the current step
                 
            # log_info(
            #     f"messages_array_to_generate_reward with multiple steps after step 1: {messages_array_to_generate_reward}"
            # )
        else:
            standard_first_user_message = (
            f"### Question:\n{question}\n\n### Solution Process:\n{now_step}"
        ) # reached only first step

            standard_first_question_in_messages_array_format, standard_first_question_corresponding_image_data_base64_list = (
                prepare_question_array_with_base64_image_strings(
                    standard_first_user_message,
                    base64_image_list,
                    interleave_image_tokens=interleave_image_tokens,
                )
            )

            log_info(f"in VisualPRM length of standard_first_question_corresponding_image_data_base64_list: {len(standard_first_question_corresponding_image_data_base64_list)}")

            messages_array_to_generate_reward += (
                standard_first_question_in_messages_array_format
            )
            

        # log_info(f"Reward model messages array: {messages_array_to_generate_reward}")
        log_info(f"Base64 image list length: {len(base64_image_list)}")
        result = self.inference_single(messages_array_to_generate_reward)

        return result

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Vision Language Model on accuracy with step level score aggregation"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the vision language model",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Tensor parallel size for VLLM",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="GPU memory utilization ratio",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        help="Data type for model weights",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required = True,
        help="Output path for results",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        required = True,
        help="Path to step traces",
    )
    parser.add_argument(
        "--num-workers", type=int, default=32, help="Number of parallel workers"
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default="You are a Math Teacher evaluating student solutions. Given a question with visual elements and a student's step-by-step solution, evaluate the mathematical correctness and logic consistency of each step. Respond with '+' if the step is correct or '-' if it is incorrect.",
        help="System prompt for the model",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to process (for testing)",
    )
    parser.add_argument(
        "--save-intermediate",
        action="store_true",
        help="Save intermediate results during processing",
    )
    parser.add_argument(
        "--tts-type",
        type=str, 
        required=True, 
        choices=["non_greedy", "step_agg"],
        help="type of test time scaling to apply"
    )
    parser.add_argument(
        "--run-datetime",
        type=str,
        required=False,
        help="datetime string to include in output filename for matching with logs"
    )

    args = parser.parse_args()

    data = load_json(args.data_path)
    # print(data[0])
    # data = data[:2] # dev mode

    model = VisualPRM(args)

    # file_name = os.path.basename(args.data_path)
    file_name = args.data_path
    # file_name = os.path.basename(args.output_path)


    for sample in tqdm(data):
        question = sample['annotation']['question']
        user_prompt = POLICY_USER_PROMPT.replace("{{QUESTION}}", question)
        image_str = sample['annotation']['image']

        if "puzzleVQA".lower() in file_name.lower() or "AlgoPuzzleVQA".lower() in file_name.lower():
            options = sample['annotation']['options']
            options_str = "\n".join(
                        [f"{chr(65 + i)}: {option}" for i, option in enumerate(options)]
                    )
            user_prompt += MCQ_SUFFIX_PROMPT_NO_TEMPLATE.replace(
                        "{{OPTIONS}}", options_str
                    )
        elif "MMMU".lower() in file_name.lower():
            # raise NotImplementedError("MMMU Data processing TODO")
            if sample['annotation']['question_type'] in [
                "multi-choice",
                "multiple-choice",
            ]:
                # options = []
                option_chars = ["A", "B", "C", "D", "E", "F", "G", "H", "I"]
                option_strs = []
                for option_char in option_chars:
                    option_val = sample['annotation'][option_char]
                    if option_val != None and not safe_isnan(option_val):
                        # print(option_val)
                        # print(type(option_val))
                        # print(safe_isnan(option_val))
                        option_strs.append(f"{option_char}: {option_val}")
                
                options_str = "\n".join(option_strs)
                user_prompt += MCQ_SUFFIX_PROMPT_NO_TEMPLATE.replace(
                        "{{OPTIONS}}", options_str
                    )
                
        elif "mathvista".lower() in file_name.lower():
            if sample['annotation']['question_type'] in [
                "multi-choice",
                "multiple-choice",
            ]:
                options = sample['annotation']['choices']
                options_str = "\n".join([f"{option}" for option in options]) + "\n"
                user_prompt += MCQ_SUFFIX_PROMPT_NO_TEMPLATE.replace(
                            "{{OPTIONS}}", options_str
                        )
        elif "mathvision".lower() in file_name.lower():
            annotation = sample["annotation"]
            question = annotation['question']
            options = ''
            if len(annotation['options']) > 0:
                assert len(annotation['options']) == 5, annotation
                if ''.join(annotation['options']) != 'ABCDE':
                    options = f"(A) {annotation['options'][0]}\n(B) {annotation['options'][1]}\n(C) {annotation['options'][2]}\n(D) {annotation['options'][3]}\n(E) {annotation['options'][4]}\n"
            # input = 'Please solve the problem step by step and put your answer in one "\\boxed{}". If it is a multiple choice question, only one letter is allowed in the "\\boxed{}".\n'+f"{question}\n{options}"
            user_prompt = ""

            user_prompt += f"Question: {question}\n"

            if options and len(options) > 0:
                user_prompt += f"Options:{options}"
                user_prompt += "Please select the correct answer from the options above.\n"
        else:
            raise ValueError(
                f"Check MCQ formatting for dataset before running: {args.data_path}"
            )

        max_score = -1
        best_idx = -1
        if len(sample['iteration_history']) < 1:
            continue
        
        for cidx, candidate in enumerate(sample['iteration_history'][0]['candidates_info']):
            if args.tts_type == "step_agg":
                steps = candidate['candidate_step'].replace("<|im_end|>", "").replace("<end_of_turn>", "").split("\n\n")
                reward = model.get_agg_reward(image_str, user_prompt, steps)

            elif args.tts_type == "non_greedy":
                solution = candidate['candidate_step'].replace("<|im_end|>", "").replace("<end_of_turn>", "")
                # Treating Entire Solution as a single step
                reward = model.get_agg_reward(image_str, user_prompt, [solution])

            else:
                raise ValueError(f"Unkown Test Time Scaling type: {args.tts_type}")

                        
            if reward > max_score:
                max_score = reward
                best_idx = cidx
            
            candidate[f"{args.tts_type}_reward"] = reward

        sample[args.tts_type] = {
            "chosen_candidate": best_idx,
            "best_reward": max_score
        }

    # Generate output filename based on input data file
    input_filename = os.path.basename(args.data_path)
    
    # Include run datetime if provided for easy matching with logs
    datetime_suffix = f"_{args.run_datetime}" if args.run_datetime else ""
    
    if input_filename.endswith('.json'):
        output_filename = input_filename.replace('.json', f'_prm_{args.tts_type}_results{datetime_suffix}.json')
    else:
        output_filename = f"{input_filename}_prm_{args.tts_type}_results{datetime_suffix}.json"
    
    output_filepath = os.path.join(args.output_path, output_filename)
    print(f"Saving results to: {output_filepath}")
    try:
        save_json(data, output_filepath)
    except Exception as e:
        print(f"Error saving results to {output_filepath}: {e}")
        raise e



if __name__ == "__main__":
    main()
        