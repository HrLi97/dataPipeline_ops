# common/vision_caption.py
"""
Vision captioning wrapper for Qwen2.5-VL.
All heavy imports are at module top as you requested.
Provides VisionCaptionModel with a simple generate_caption API.
"""

import os
import json
from typing import List, Dict, Any, Tuple

import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info  # keep this import at top per your rule

# You may tune these min/max based on available GPU memory / expected image sizes
DEFAULT_MIN_PIXELS = 16 * 28 * 28
DEFAULT_MAX_PIXELS = 20480 * 28 * 28


class VisionCaptionModel:
    """
    Encapsulates processor + model for Qwen2.5-VL inference.
    Usage:
        vc = VisionCaptionModel(model_id, min_pixels=..., max_pixels=..., torch_dtype=..., device_map='auto')
        text = vc.generate_caption(placeholders, question, max_new_tokens=128)
    """

    def __init__(
        self,
        model_id: str,
        min_pixels: int = DEFAULT_MIN_PIXELS,
        max_pixels: int = DEFAULT_MAX_PIXELS,
        torch_dtype: str = "bfloat16",
        device_map: str = "auto",
        attn_implementation: str = "flash_attention_2",
    ):
        self.model_id = model_id
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.torch_dtype = torch_dtype
        self.device_map = device_map
        self.attn_implementation = attn_implementation

        # Load processor and model at initialization (heavy)
        self.processor = AutoProcessor.from_pretrained(self.model_id, min_pixels=self.min_pixels, max_pixels=self.max_pixels)
        # model dtype selection
        torch_dtype_val = getattr(torch, self.torch_dtype) if isinstance(self.torch_dtype, str) and hasattr(torch, self.torch_dtype) else torch.bfloat16
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=torch_dtype_val,
            device_map=self.device_map,
            attn_implementation=self.attn_implementation,
        )

    def _build_messages(self, placeholders: List[Dict[str, Any]], question: str) -> List[Dict[str, Any]]:
        """
        Build messages list used by processor.apply_chat_template.
        placeholders: list of {"type":"image","image":path,"image_id":id} ... with the last image being target (no image_id)
        question: the textual instruction (string)
        """
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": [*placeholders, {"type": "text", "text": question}]},
        ]
        return messages

    def generate_caption(self, placeholders: List[Dict[str, Any]], question: str, max_new_tokens: int = 128) -> str:
        """
        Run the whole inference flow and return the text (string).
        """
        messages = self._build_messages(placeholders, question)

        # prepare text template (tokenize=False per original)
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        # extract image/video tensors via your helper
        image_inputs, video_inputs = process_vision_info(messages)

        # prepare processor inputs
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        # move to GPU (model.device is already placed by device_map="auto")
        # but inputs must reside on same device as model
        try:
            # choose device of model first param
            device = next(self.model.parameters()).device
        except StopIteration:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        inputs = inputs.to(device)

        # inference
        generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)

        # remove input ids prefix
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]

        # decode
        output_texts = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        # return first result (original code used output_text[0])
        return output_texts[0] if output_texts else ""
