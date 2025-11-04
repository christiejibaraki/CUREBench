"""
Bio-Medical AI Competition Starter Kit

A simple framework for evaluating models on bio-medical datasets.
Perfect for getting started quickly in the competition.

Key Features:
- Easy model loading (ChatGPT, GPT-OSS-20B, Local models, Custom models)
- Simple dataset loading
- Automatic evaluation and scoring
- Submission file generation

Usage:
    framework = CompetitionKit()
    framework.load_model("gpt-4o-mini")
    results = framework.evaluate("quick_test")
    framework.sa        elif question_type == "open_ended":
            # For open-ended, only return response, use NOTAVALUE for choice to avoid empty string issues
            prediction["choice"] = "NOTAVALUE"  # Use NOTAVALUE instead of empty string to avoid NULL validation issues
            prediction["open_ended_answer"] = response.strip()ubmission(results, "my_submission.json")
"""

import json
import os
import sys
import logging
import argparse
import torch# Import for stopping criteria
import torch
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from tqdm import tqdm
from abc import ABC, abstractmethod
from transformers import StoppingCriteria, StoppingCriteriaList
import csv

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


system_identity = """
You are an expert medical assistant specializing in answering questions.

**Your communication MUST strictly adhere to the Harmony channels:**
1.  **analysis:** Use this for all internal Chain-of-Thought (CoT), clinical reasoning, and factual processing. This content is for internal use only.
2.  **final:** Use this channel for the final output intended for the user.

**Output Rule is Conditional:**
* **If the input is a Multiple-Choice Question (MCQ):** Your output MUST be a single, valid JSON object containing only the selected answer letter.
    * **Format:** `{"answer": "<LETTER>"}` (e.g., `{"answer": "A"}`)
* **If the input is an Open-Ended Question:** Your output MUST be a detailed, coherent narrative response.

**Instruction:** Generate a complete response. The final output must use the 'final' channel and adhere to the conditional format rule."""


stop_sequences = [
    # Use very conservative stop sequences to avoid breaking Harmony token structure
    # Let Harmony's native stopping handle most cases
    # Only catch extreme repetition patterns that clearly indicate infinite loops
]

unsloth_model_name = "unsloth/gpt-oss-20b"
unsloth_developer_instructions = "You are a medical expert for drug decision-making and treatment planning. During your analysis, determine if the question is multiple-choice (MC) or open-ended (OE). For MC questions, your final response must be ONLY the correct LETTER. For OE questions, provide a succinct, single sentence response."
unsloth_lora_adapters = "cibaraki/medical-reasoning-gpt-oss-20b"


class CustomStopStringCriteria(StoppingCriteria):
    """Custom criteria to stop generation only in extreme repetition cases."""
    def __init__(self, stop_strings: List[str], tokenizer):
        super().__init__()
        self.stop_strings = stop_strings
        self.tokenizer = tokenizer
        self.call_count = 0

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> bool:
        self.call_count += 1

        # Only check every 10 calls to reduce overhead and avoid breaking mid-token
        if self.call_count % 10 != 0:
            return False

        # Use a large lookback to catch patterns but avoid breaking token structure
        lookback_window = 300
        last_tokens = input_ids[0, -lookback_window:].tolist()
        # Decode with skip_special_tokens=True so we can actually see the text content
        # (Harmony uses special tokens like <|start|>assistant<|channel|>final that would be invisible otherwise)
        text = self.tokenizer.decode(last_tokens, skip_special_tokens=True)

        # Only stop if we see EXTREME repetition (same 100+ char block repeated)
        if len(text) > 200:
            mid = len(text) // 2
            first_half = text[:mid]
            second_half = text[mid:]
            # If the second half starts with the same content as first half, it's looping
            if len(first_half) > 100 and second_half.startswith(first_half[:100]):
                return True

        # Check for custom stop strings only if provided
        for stop_str in self.stop_strings:
            if stop_str and stop_str in text:
                return True

        return False


@dataclass
class EvaluationResult:
    """Simple container for evaluation results"""
    dataset_name: str
    model_name: str
    accuracy: float
    correct_predictions: int
    total_examples: int
    predictions: List[Dict]  # Changed from List[str] to List[Dict]
    reasoning_traces: List[str] = None  # Add reasoning traces
    details: Optional[Dict] = None


# Model Classes
class BaseModel(ABC):
    """Abstract base class for all models"""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None

    @abstractmethod
    def load(self, **kwargs):
        """Load the model"""
        pass

    @abstractmethod
    def inference(self, prompt: str, max_tokens: int = 1024) -> Tuple[str, List[Dict]]:
        """Run inference on the model

        Returns:
            Tuple of (response, messages) where messages is the complete conversation history
        """
        pass


class UnslothGPTOSS20BModel(BaseModel):
    """Wrapper for unsloth gpt-oss-20b model (including fine-tuned version)"""

    def __init__(
        self,
        model_name: str,
        quantization: bool = True,       # auto | fp16 | bf16 | 8bit
        reasoning_lvl: str = "low",       # low | medium | high
        developer_instructions: str = None,  # optional developer message
        lora_adapters: str = None  # optional fine-tuned adapters
    ):
        super().__init__(model_name)
        self.quantization = quantization
        self.model = None
        self.tokenizer = None
        self.enc = None
        self.reasoning_lvl = reasoning_lvl
        self.developer_instructions = developer_instructions
        self.lora_adapters = lora_adapters

    def load(self, **kwargs):
        from unsloth import FastLanguageModel

        max_seq_length = 1024
        dtype = None

        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.model_name,
            dtype=dtype,  # None for auto detection
            max_seq_length=max_seq_length,  # Choose any for long context!
            load_in_4bit=self.quantization,  # 4 bit quantization to reduce memory
            full_finetuning=False,  # [NEW!] We have full finetuning now!
            # token = "hf_...", # use one if using gated models
        )

        if self.lora_adapters:
            self.model.load_adapter(self.lora_adapters)

    def inference(self, prompt: str, max_tokens: int = 512, temperature: float = 0.3, top_p: float = 0.9) -> Tuple[str, List[Dict]]:
        import re

        #### Generate the output ####
        messages = []
        if self.developer_instructions:
            messages.append({"role": "developer", "content": self.developer_instructions})
        messages.append({"role": "user", "content": prompt})

        # Apply chat template (from unsloth)
        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            reasoning_effort=self.reasoning_lvl
        ).to("cuda")
        # Generate output
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=512
        )
        # Decode the output, leaving special tokens
        full_response = self.tokenizer.decode(output_ids[0], skip_special_tokens=False)

        #### Extract final reaspone and reasoning trace ####
        # Regex Patterns
        # pattern to find all assistant messages
        ASSISTANT_MESSAGE_PATTERN = re.compile(
            r'<\|start\|>assistant(?P<header>.+?)<\|message\|>(?P<content>.*?)(?=<\|end\|>|<\|start\|>|<\|return\|>|$)',
            re.DOTALL | re.IGNORECASE
        )
        # pattern to extract the channel from the header
        CHANNEL_PATTERN = re.compile(r'<\|channel\|>(?P<channel>\w+)')

        reasoning_trace = []
        final_response = "ERROR: Final response not found."

        # Find all assistant messages in the full response
        assistant_messages = ASSISTANT_MESSAGE_PATTERN.findall(full_response)

        for header, content in assistant_messages:
            # Extract the channel(s) from the header
            channel_matches = CHANNEL_PATTERN.findall(header)

            message_dict = {
                "role": "assistant",
                "channel": channel_matches[-1] if channel_matches else "unknown",
                "content": content.strip()
            }

            # Add to the reasoning trace
            reasoning_trace.append(message_dict)

            # Check if this is the final message
            if message_dict['channel'] == 'final':
                final_response = message_dict['content']

        return final_response, reasoning_trace


class GPTOSS20BModel(BaseModel):
    """GPT-OSS-20B wrapper"""

    def __init__(
        self,
        model_name: str,
        quantization: str = "auto",          # auto | fp16 | bf16 | 8bit
        reasoning_lvl: str = "low",       # low | medium | high
        system_identity: str = None,         # optional system override
        developer_instructions: str = None,  # optional developer message
    ):
        super().__init__(model_name)
        self.quantization = quantization
        self.model = None
        self.tokenizer = None
        self.reasoning_lvl = reasoning_lvl
        self.system_identity = system_identity
        self.developer_instructions = developer_instructions

    def load(self, **kwargs):
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        import torch
        from openai_harmony import load_harmony_encoding, HarmonyEncodingName

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        if self.quantization == "fp16":
            torch_dtype = torch.float16
            quant_config = None
        elif self.quantization == "bf16":
            torch_dtype = torch.bfloat16
            quant_config = None
        elif self.quantization == "8bit":
            torch_dtype = torch.bfloat16
            quant_config = None
        else:
            # this will automatically use MXFP4 weights.
            torch_dtype = "auto"
            quant_config = None

        model_kwargs = {"torch_dtype": torch_dtype, "device_map": "auto", **kwargs}
        if quant_config is not None:
            model_kwargs["quantization_config"] = quant_config


    def inference(self, prompt: str, max_tokens: int = 512, temperature: float = 0.3, top_p: float = 0.9,
                  builtin_tools: Optional[List[str]] = None, tools: Optional[List[dict]] = None,
                  stop_strings: Optional[List[str]] = stop_sequences) -> Tuple[str, List[Dict]]:

        from openai_harmony import Role
        import logging
        from transformers import AutoTokenizer
        import torch

        # Build message list
        messages = []
        if self.system_identity or self.reasoning_lvl:
            sys_content = ""
            if self.system_identity:
                sys_content += self.system_identity + "\n"
            sys_content += f"Reasoning: {self.reasoning_lvl}."
            messages.append({"role": "system", "content": sys_content})

        if self.developer_instructions:
            messages.append({"role": "developer", "content": self.developer_instructions})

        messages.append({"role": "user", "content": prompt})

        # Apply Hugging Face chat template with fallback
        try:
            input_ids = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
                reasoning_effort=self.reasoning_lvl,
                model_identity=self.system_identity
                    or system_identity,
                builtin_tools=builtin_tools,
                tools=tools,
            ).to(self.model.device)
        except Exception as e:
            logging.warning(
                f"[WARN] Custom chat_template in {self.model_name} failed "
                f"({type(e).__name__}: {e}). Falling back to base GPT-OSS template."
            )
            # Reload base tokenizer for Harmony
            base_tok = AutoTokenizer.from_pretrained("openai/gpt-oss-20b")
            self.tokenizer.chat_template = base_tok.chat_template
            input_ids = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
                reasoning_effort=self.reasoning_lvl,
                model_identity=self.system_identity
                    or system_identity,
                builtin_tools=builtin_tools,
                tools=tools,
            ).to(self.model.device)

        # Build Stopping Criteria List ---
        # 1. Start with the official stop tokens from the Harmony encoding
        stopping_criteria = StoppingCriteriaList()

        # 2. Add custom string-based stop criteria for the observed noise
        if stop_strings:
            criteria = CustomStopStringCriteria(stop_strings, self.tokenizer)
            stopping_criteria.append(criteria)

        outputs = self.model.generate(
            input_ids,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_tokens,
            do_sample=(temperature>0),
            eos_token_id=None if not self.enc else self.enc.stop_tokens()[-1],
            stopping_criteria=stopping_criteria if stopping_criteria else None,
            repetition_penalty=1.3,  # Increased from 1.2 to further discourage repetition
            no_repeat_ngram_size=3,  # Prevent repetition of 3-grams
        )
        # Parse Harmony messages
        gen_tokens = outputs[0][input_ids.shape[-1]:].tolist()

        try:
            parsed = self.enc.parse_messages_from_completion_tokens(gen_tokens, role=Role.ASSISTANT)
            reasoning_trace = [msg.to_dict() for msg in parsed]

            # Prefer "final" channel
            finals = [msg for msg in parsed if msg.to_dict().get("channel") == "final"]
            if finals:
                final_response = "".join(c.text for c in finals[-1].content if hasattr(c, "text"))
            else:
                # Fallback: take last assistant message, but strip to short answer
                final_response = "".join(c.text for c in parsed[-1].content if hasattr(c, "text"))

        except Exception as e:
            # Harmony parsing failed - likely due to incomplete token sequence
            logging.warning(f"[Harmony parse error] {e} - Using raw decode fallback")

            # Decode the raw text
            text = self.tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

            # Post-process: truncate at first sign of repetition
            import re
            # Find the first complete JSON answer
            json_match = re.search(r'\{"answer"\s*:\s*"([A-E])"\}', text)
            if json_match:
                # Find where this match ends
                match_end = json_match.end()
                # Look for repetition after the answer
                after_answer = text[match_end:]
                # If we see the same question or analysis starting again, truncate
                repeat_markers = ["The user asks:", "The task: The user wrote", "assistantfinalanalysis"]
                for marker in repeat_markers:
                    if marker in after_answer:
                        # Truncate at the repetition
                        text = text[:match_end + after_answer.index(marker)]
                        break

                final_response = f'{{"answer": "{json_match.group(1)}"}}'
            else:
                # Look for assistantfinal content
                final_match = re.search(r'assistantfinal(.+?)(?:analysis|assistant|$)', text, re.DOTALL)
                if final_match:
                    final_response = final_match.group(1).strip()
                else:
                    # Last resort: use the full text but truncate obvious loops
                    # If we see "The user asks:" or "The task:" appearing multiple times, keep only first occurrence
                    for marker in ["The user asks:", "The task: The user"]:
                        parts = text.split(marker)
                        if len(parts) > 2:  # Appears more than once
                            text = parts[0] + marker + parts[1]
                            break
                    final_response = text

            reasoning_trace = [{"role": "assistant", "content": text}]

        return final_response.strip(), reasoning_trace


class CompetitionKit:
    """
    Simple competition framework - everything you need in one class!
    """

    def __init__(self, config_path: str = None):
        """
        Initialize the competition kit

        Args:
            output_dir: Directory to save results and submissions
            config_path: Path to configuration file containing dataset configs
        """
        self.model = None
        self.model_name = None

        self.config = json.load(open(config_path, 'r')) if config_path else {}

        self.output_dir = self.config.get('output_dir', 'results')
        self.metadata_path = None
        self.metadata_filename = None
        self.csv_path = None
        self.csv_filename = None

        # Load dataset configurations from config file or use defaults
        self.datasets = self._load_dataset_configs(self.config)

    def load_model(self, model_name: str, model_type: str = "auto", **kwargs):
        """
        Load a model for evaluation

        Args:
            model_name: Name/path of the model (e.g., "gpt-4o-mini", "meta-llama/Llama-2-7b-chat-hf")
            model_type: Type of model ("chatgpt", "local", "custom", "auto" for auto-detection)
            **kwargs: Additional model configuration
        """
        self.model_name = model_name

        # Auto-detect model type if not specified
        if model_type == "auto":
            model_type = self._detect_model_type(model_name)

        logger.info(f"Loading model: {model_name} (type: {model_type})")

        if model_type == "gpt-oss-20b":
            self.model = GPTOSS20BModel("openai/gpt-oss-20b", system_identity=system_identity)
        elif model_type == "unsloth/gpt-oss-20b":
            self.model = UnslothGPTOSS20BModel(model_name=unsloth_model_name,
                                               developer_instructions=unsloth_developer_instructions,
                                               lora_adapters=unsloth_lora_adapters)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Load the model
        self.model.load(**kwargs)

    def _load_dataset_configs(self, config) -> Dict:
        """
        Load dataset configurations from config file or return defaults

        Args:
            config: Configuration dictionary

        Returns:
            Dictionary of dataset configurations
        """
        if not config:
            print("No config provided, existing.")
            exit(1)

        # Check if config has a single dataset configuration
        if 'dataset' in config:
            dataset_config = config['dataset']
            dataset_name = dataset_config.get('dataset_name', 'treatment')
            # Create a dictionary with the dataset name as key
            return {dataset_name: dataset_config}
        else:
            # If no dataset in config, return defaults
            print("No config found, existing.")
            exit(1)

    def _detect_model_type(self, model_name: str) -> str:
        """Auto-detect model type based on model name"""
        if "gpt-oss-20b" in model_name.lower():
            if "unsloth" in model_name.lower():
                return "unsloth/gpt-oss-20b"
            else:
                return "gpt-oss-20b"
        if any(name in model_name.lower() for name in ["gpt", "chatgpt", "openai", 'o1', 'o3', 'o4']):
            return "chatgpt"
        else:
            return "local"

    def evaluate(self, dataset_name: str, csv_writer: csv.DictWriter,
                 start_index: int = 0, subset_size: int = None) -> EvaluationResult:
        """
        Evaluate model on a dataset

        Args:
            dataset_name: Name of dataset to evaluate on
            csv_writer: csv.DictWriter for streaming output
            start_index: int index of example at which to begin eval
            subset_size: int number of examples to evaluate

        Returns:
            EvaluationResult object with scores and predictions
        """
        if not self.model:
            raise ValueError("No model loaded. Call load_model() first.")

        if dataset_name not in self.datasets:
            raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(self.datasets.keys())}")

        dataset_config = self.datasets[dataset_name]
        logger.info(f"Evaluating on {dataset_name}: {dataset_config['description']}")

        # Load dataset
        dataset = self._load_dataset(dataset_config)

        # Store dataset examples for later use in save_submission
        self._last_dataset_examples = dataset

        if subset_size is not None and subset_size > 0:
            last_index = (start_index + subset_size)
            dataset = dataset[start_index:last_index]
            logger.info(f"Starting evaluation on example index: {start_index}. Last index: {last_index}")
            logger.info(f"Subset size applied: {len(dataset)} examples")

        # Run evaluation
        predictions = []
        reasoning_traces = []  # Store reasoning traces
        total_count = len(dataset)
        # Track accuracy only for non-open-ended questions
        accuracy_correct_count = 0
        accuracy_total_count = 0

        logger.info(f"Running evaluation on {total_count} examples...")
        for i, example in enumerate(tqdm(dataset, desc="Evaluating")):

            prediction = None
            reasong_trace = None

            try:
                # Get prediction and reasoning trace
                prediction, reasoning_trace = self._get_prediction_with_trace(example)

                # Check if correct based on question type
                is_correct = False
                question_type = example["question_type"]
                expected_answer = example.get("answer")

                if question_type == "multi_choice" or question_type == "open_ended_multi_choice":
                    # For multiple choice, compare the choice field
                    if expected_answer !='':
                        is_correct = prediction["choice"] == expected_answer
                    else:
                        is_correct = False
                    # Count for accuracy calculation (exclude open_ended)
                    accuracy_total_count += 1
                    if is_correct:
                        accuracy_correct_count += 1
                elif question_type == "open_ended":
                    # For open-ended, compare the open_ended_answer field but don't count in accuracy, we have internal evaluation for open-ended questions
                    if expected_answer !='':
                        is_correct = prediction["open_ended_answer"] == expected_answer
                    else:
                        is_correct = False

                # Log progress
                if (i + 1) % 10 == 0:
                    # csv_writer.flush()
                    current_acc = accuracy_correct_count / accuracy_total_count if accuracy_total_count > 0 else 0.0
                    logger.info(f"Progress: {i+1}/{total_count}, Accuracy: {current_acc:.2%} (excluding open-ended)")

            except Exception as e:
                logger.error(f"Error processing example {i}: {e}", exc_info=True)
                error_prediction = {
                    "choice": "NOTAVALUE",  # Use NOTAVALUE instead of empty string
                    "open_ended_answer": "Error"
                }
                prediction = error_prediction
                reasoning_trace = "Error occurred during inference"

            # add results to list and stream output
            predictions.append(prediction)
            reasoning_traces.append(reasoning_trace)
            self._write_prediction_to_csv(csv_writer,
                              prediction, reasoning_trace,
                              example, i)

        # Calculate final accuracy (excluding open-ended questions)
        accuracy = accuracy_correct_count / accuracy_total_count if accuracy_total_count > 0 else 0.0

        result = EvaluationResult(
            dataset_name=dataset_name,
            model_name=self.model_name,
            accuracy=accuracy,
            correct_predictions=accuracy_correct_count,  # Use accuracy-specific count
            total_examples=accuracy_total_count,  # Use accuracy-specific count
            predictions=predictions,
            reasoning_traces=reasoning_traces  # Include reasoning traces
        )

        logger.info(f"Evaluation completed: {accuracy:.2%} accuracy ({accuracy_correct_count}/{accuracy_total_count}) - excluding open-ended questions")
        logger.info(f"Total examples processed: {total_count} (including {total_count - accuracy_total_count} open-ended questions)")

        return result

    def _load_dataset(self, dataset_config: Dict) -> List[Dict]:
        """Load dataset based on configuration"""
        from dataset_utils import build_dataset
        from torch.utils.data import DataLoader

        # Build dataset
        dataset = build_dataset(
            dataset_config.get("dataset_path"),
        )

        # Convert to list of dictionaries for easier processing
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        dataset_list = []

        for batch in dataloader:
            question_type = batch[0][0]

            if question_type == "multi_choice":
                dataset_list.append({
                    "question_type": batch[0][0],
                    "id": batch[1][0],
                    "question": batch[2][0],
                    "answer": batch[3][0],
                })
            elif question_type == "open_ended_multi_choice":
                dataset_list.append({
                    "question_type": batch[0][0],
                    "id": batch[1][0],
                    "question": batch[2][0],
                    "answer": batch[3][0],
                    "meta_question": batch[4][0],
                })
            elif question_type == "open_ended":
                dataset_list.append({
                    "question_type": batch[0][0],
                    "id": batch[1][0],
                    "question": batch[2][0],
                    "answer": batch[3][0],
                })

        return dataset_list

    def _get_prediction_with_trace(self, example: Dict) -> Tuple[Dict, str]:
        """Get model prediction and reasoning trace for a single example"""
        question = example["question"]
        question_type = example["question_type"]

        prompt = question

        # Get model response and messages using the model's inference method
        response, reasoning_trace = self.model.inference(prompt)

        # Initialize prediction dictionary
        prediction = {
            "choice": "",  # Use empty string instead of None
            "open_ended_answer": ""  # Use empty string instead of None
        }

        # Extract answer from response
        if question_type == "multi_choice":
            # For multiple choice, extract the letter
            choice = self._extract_multiple_choice_answer(response)
            # Ensure choice is never None or NULL
            prediction["choice"] = choice if choice and str(choice).upper() not in ['NONE', 'NULL'] else ""
            prediction["open_ended_answer"] = response.strip()  # Keep full response too
        elif question_type == "open_ended_multi_choice":
            # First get the detailed response
            prediction["open_ended_answer"] = response.strip()

            # Then use meta question to get choice, if available
            if "meta_question" in example:
                meta_prompt = f"{example['meta_question']}Agent's answer: {response.strip()}\n\nMulti-choice answer:"
                meta_response, meta_reasoning = self.model.inference(meta_prompt)
                # Combine reasoning traces
                reasoning_trace += meta_reasoning
                # Extract the letter choice
                choice = self._extract_multiple_choice_answer(meta_response)
                # Ensure choice is never None or NULL
                prediction["choice"] = choice if choice and str(choice).upper() not in ['NONE', 'NULL'] else ""
            else:
                # If no meta_question, try to extract choice directly from the response
                choice = self._extract_multiple_choice_answer(response)
                # Ensure choice is never None or NULL
                prediction["choice"] = choice if choice and str(choice).upper() not in ['NONE', 'NULL'] else ""
        elif question_type == "open_ended":
            # For open-ended, only return response, use N/A for choice to avoid empty string issues
            prediction["choice"] = "NOTAVALUE" # Use N/A instead of empty string to avoid NULL validation issues
            prediction["open_ended_answer"] = response.strip()

        return prediction, reasoning_trace

    def _extract_multiple_choice_answer(self, response: str) -> str:
        """Extract letter answer from model response"""
        if not response or response is None:
            return ""

        response = response.strip().upper()

        # Look for letter at the beginning
        if response and response[0] in ['A', 'B', 'C', 'D', 'E']:
            return response[0]

        # Look for "The answer is X" patterns
        import re
        patterns = [
            r"(?:answer is|answer:|is)\s*([ABCDE])",
            r"([ABCDE])\)",
            r"\b([ABCDE])\b"
        ]

        for pattern in patterns:
            match = re.search(pattern, response)
            if match:
                return match.group(1)

        # Default to empty string if nothing found (to avoid None values in CSV)
        return ""

    def _write_prediction_to_csv(self, csv_writer, prediction, reasoning_trace, example, index):
        """
        Write a single prediction to CSV

        Args:
            csv_writer: csv.DictWriter for writing output
            prediction: dict containing model "choice" and "open_ended_answer"
            reasoning_trace: List of reasoning message dictionaries
            example: dict containing original dataset example
            index: integer index of current example
        """
        example_id = str(example.get("id", str(index)) or f"unknown_{index}")

        # Clean up text fields to avoid CSV formatting issues
        prediction_text = prediction.get("open_ended_answer", "") or ""  # Ensure not None
        if not prediction_text or prediction_text.strip() == "":
            prediction_text = "No prediction available"

        # Ensure choice is clean and never NULL
        choice_raw = prediction.get("choice", "")
        if choice_raw is None or str(choice_raw).upper() in ['NULL', 'NONE', 'NAN']:
            choice_clean = "NOTAVALUE"  # Use NOTAVALUE instead of empty string
            logger.warning(
                f"Found NULL-like or empty choice for row {example_id}: '{choice_clean}' - replacing with NOTAVALUE")
        elif str(choice_raw).strip() == "":
            choice_clean = "NOTAVALUE"  # Replace empty strings with NOTAVALUE to avoid NULL validation issues
        else:
            choice_clean = str(choice_raw).strip()

        # Ensure reasoning trace is not null
        reasoning_trace = json.dumps(reasoning_trace)
        if not reasoning_trace or reasoning_trace == "null" or reasoning_trace.strip() == "":
            reasoning_trace = "No reasoning available"

        csv_writer.writerow({
            "id": example_id,
            "prediction": str(prediction_text),
            "choice": str(choice_clean),
            "reasoning": str(reasoning_trace)
        })

    def initialize_output(self, filename: str = "submission.csv",
                          metadata: Dict = None, config_path: str = None,
                          args: argparse.Namespace = None):
        """
        Initialize output directory and file

        Args:
            filename: Output CSV filename (will be used for CSV inside zip)
            metadata: User-provided metadata dictionary containing model info, track, etc.
            config_path: Path to configuration file containing metadata
            args: Command line arguments containing metadata
        """
        # if output dir doesn't exist, create it
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            print(f"Directory {self.output_dir} created successfully.")

        # Open CSV file for streaming results
        csv_path = os.path.join(self.output_dir, filename)
        csv_exists = os.path.exists(csv_path)
        file_mode = 'a' if csv_exists else 'w'
        csv_file = open(csv_path, file_mode, newline='', encoding='utf-8')
        csv_writer = csv.DictWriter(csv_file, fieldnames=['id', 'prediction', 'choice', 'reasoning'])
        if not csv_exists:
            csv_writer.writeheader()

        # Get metadata from various sources with priority order
        metadata = self.get_metadata(config_path, args, metadata)
        # Create metadata JSON file
        metadata_filename = "meta_data.json"
        metadata_path = os.path.join(self.output_dir, metadata_filename)
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        self.metadata_path = metadata_path
        self.metadata_filename = metadata_filename
        self.csv_path = csv_path
        self.csv_filename = filename
        logger.info(f"Initialized CSV for streaming: {csv_path}")
        return csv_writer, csv_file

    def finalize_output(self, csv_file, results: List[EvaluationResult]):
        """
        Close CSV file, zip submission csv with metadata, and provide summary

        Args:
            csv_file: csv file containing eval results
            results: List of EvaluationResult objects from evaluate()
        """
        import zipfile
        # Close csv file
        csv_file.close()

        # Create ZIP file with CSV and metadata
        zip_filename = self.csv_filename.replace('.csv', '.zip')
        zip_path = os.path.join(self.output_dir, zip_filename)
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Add CSV file to zip
            zipf.write(self.csv_path, self.csv_filename)
            # Add metadata JSON to zip
            zipf.write(self.metadata_path, self.metadata_filename)

        # Calculate and log overall accuracy
        total_correct = sum(r.correct_predictions for r in results)
        total_examples = sum(r.total_examples for r in results)
        overall_accuracy = total_correct / total_examples if total_examples > 0 else 0.0

        logger.info(f"CSV submission saved to: {self.csv_path}")
        logger.info(f"Metadata saved to: {self.metadata_path}")
        logger.info(f"Submission package saved to: {zip_path}")
        logger.info(
            f"Overall accuracy (excluding open-ended questions): {overall_accuracy:.2%} ({total_correct}/{total_examples})")

    def list_datasets(self):
        """List available datasets"""
        print("Available Datasets:")
        print("-" * 50)
        for name, config in self.datasets.items():
            print(f"  {name}: {config['description']}")

    def load_metadata_from_config(self, config_path: str) -> Dict:
        """
        Load metadata from configuration file

        Args:
            config_path: Path to configuration file (JSON or YAML)

        Returns:
            Metadata dictionary
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        _, ext = os.path.splitext(config_path)

        with open(config_path, 'r') as f:
            if ext.lower() in ['.json']:
                config = json.load(f)
            elif ext.lower() in ['.yaml', '.yml']:
                try:
                    import yaml
                    config = yaml.safe_load(f)
                except ImportError:
                    raise ImportError("PyYAML is required for YAML config files. Install with: pip install PyYAML")
            else:
                raise ValueError(f"Unsupported config file format: {ext}")

        # Extract metadata from config
        metadata = config.get('metadata', config.get('meta_data', {}))

        # Validate required fields
        required_fields = ['model_name', 'track', 'base_model_type', 'base_model_name', 'dataset']
        for field in required_fields:
            if field not in metadata:
                logger.warning(f"Required metadata field '{field}' not found in config")

        return metadata

    def parse_metadata_from_args(self, args: argparse.Namespace) -> Dict:
        """
        Parse metadata from command line arguments

        Args:
            args: Parsed command line arguments

        Returns:
            Metadata dictionary
        """
        metadata = {}

        # Map argument names to metadata fields
        arg_mapping = {
            'model_name': 'model_name',
            'model_type': 'model_type',
            'track': 'track',
            'base_model_type': 'base_model_type',
            'base_model_name': 'base_model_name',
            'dataset': 'dataset',
            'additional_info': 'additional_info'
        }

        for arg_name, meta_field in arg_mapping.items():
            if hasattr(args, arg_name) and getattr(args, arg_name) is not None:
                metadata[meta_field] = getattr(args, arg_name)

        return metadata

    def get_metadata(self, config_path: str = None, args: argparse.Namespace = None,
                    fallback_metadata: Dict = None) -> Dict:
        """
        Get metadata from various sources with priority order:
        1. Command line arguments (highest priority)
        2. Configuration file
        3. Fallback metadata provided
        4. Default metadata (lowest priority)

        Args:
            config_path: Path to configuration file
            args: Parsed command line arguments
            fallback_metadata: Fallback metadata dictionary

        Returns:
            Final metadata dictionary
        """
        # Start with default metadata
        metadata = {
            "model_name": self.model_name or "unknown",
            "model_type": type(self.model).__name__ if self.model else "Unknown",
            "track": "internal_reasoning",
            "base_model_type": "API",
            "base_model_name": self.model_name or "unknown",
            "dataset": "unknown",
            "additional_info": "Generated using eval_framework"
        }

        # Override with fallback metadata if provided
        if fallback_metadata:
            metadata.update(fallback_metadata)

        # Override with config file metadata if provided
        if config_path:
            try:
                config_metadata = self.load_metadata_from_config(config_path)
                metadata.update(config_metadata)
                logger.info(f"Loaded metadata from config file: {config_path}")
            except Exception as e:
                logger.warning(f"Failed to load config file {config_path}: {e}")

        # Override with command line arguments if provided (highest priority)
        if args:
            arg_metadata = self.parse_metadata_from_args(args)
            metadata.update(arg_metadata)
            if arg_metadata:
                logger.info(f"Applied metadata from command line arguments")

        return metadata


def create_metadata_parser() -> argparse.ArgumentParser:
    """
    Create command line argument parser for metadata

    Returns:
        ArgumentParser with metadata-related arguments
    """
    parser = argparse.ArgumentParser(description='Evaluation Framework with Metadata Support')

    # Model information
    parser.add_argument('--model-name', type=str, help='Name of the model')
    parser.add_argument('--model-type', type=str, help='Type of model wrapper')
    parser.add_argument('--base-model-name', type=str, help='Name of the base model')
    parser.add_argument('--base-model-type', type=str, choices=['API', 'OpenWeighted'],
                       help='Type of base model (API or OpenWeighted)')

    # Track information
    parser.add_argument('--track', type=str, choices=['internal_reasoning', 'agentic_reasoning'],
                       default='internal_reasoning', help='Competition track')

    # Dataset and submission info
    parser.add_argument('--dataset', type=str, help='Dataset name')
    parser.add_argument('--additional-info', type=str, help='Additional information about the submission')

    # Configuration file
    parser.add_argument('--config', type=str, help='Path to configuration file (JSON or YAML)')

    # Output settings
    parser.add_argument('--output-dir', type=str, default='competition_results',
                       help='Output directory for results')
    parser.add_argument('--output-file', type=str, default='submission.csv',
                       help='Output CSV filename for submission (will be packaged in zip)')

    # Evaluation settings
    parser.add_argument('--start-index', type=int, help='Set example index to stat evaluation')
    parser.add_argument('--subset-size', type=int, help='Limit evaluation to N examples')

    return parser


def load_config_file(config_path):
    """Load configuration from JSON file"""
    if not os.path.exists(config_path):
        print(f"❌ Error: Configuration file not found: {config_path}")
        sys.exit(1)

    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ Error loading config file {config_path}: {e}")
        sys.exit(1)


def load_and_merge_config(args):
    """Load config file and merge values into args. Command line args take precedence."""
    if not args.config:
        return args

    config = load_config_file(args.config)

    # First, handle the metadata section specially - merge its contents directly
    if 'metadata' in config:
        metadata = config['metadata']
        for key, value in metadata.items():
            if not hasattr(args, key) or getattr(args, key) is None:
                setattr(args, key, value)

    # Then handle all other config values, flattening nested structures
    def add_config_to_args(config_dict, prefix=''):
        for key, value in config_dict.items():
            if key in ['metadata', 'dataset']:  # Skip metadata and dataset as we handle them specially
                continue
            attr_name = f"{prefix}_{key}" if prefix else key
            if isinstance(value, dict):
                add_config_to_args(value, attr_name)
            elif not hasattr(args, attr_name) or getattr(args, attr_name) is None:
                setattr(args, attr_name, value)

    add_config_to_args(config)
    return args
