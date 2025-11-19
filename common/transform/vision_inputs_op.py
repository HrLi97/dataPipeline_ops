"""
把 messages (placeholders + question text) 转为模型可接受的 inputs:
- 调用 process_vision_info(messages) -> image_inputs, video_inputs
- 使用 processor.apply_chat_template 得到 text
- 使用 processor(text=..., images=..., videos=...) 得到 tensors
把最终的 tensors(CPU/记忆)放到 item['inputs']，供生成算子使用。
"""
from ..base_ops import BaseOps
from qwen_vl_utils import process_vision_info
import torch

class VisionInputsOp(BaseOps):
    def __init__(self, device: str = "cuda", **kwargs):
        self.device = device

    def predict(self, item: dict) -> dict:
        processor = item.get("processor")
        if processor is None:
            item['vision_inputs_error'] = "no processor"
            return item

        placeholders = item.get("placeholders", [])
        question = item.get("question_text", "")
        # messages 格式与原脚本保持一致
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": [*placeholders, {"type": "text", "text": question}]}
        ]

        # 1. 构造 text via processor.apply_chat_template
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        # 2. vision / video inputs via process_vision_info（来自 qwen_vl_utils）
        image_inputs, video_inputs = process_vision_info(messages)

        # 3. processor() -> returns dict of tensors
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        )

        # 将 tensors 放到 device（caller 需要保证 device 可用）
        try:
            inputs = inputs.to(self.device)
        except Exception:
            # 若 inputs 没有 .to（某些版本返回 dict of tensors）
            for k, v in inputs.items():
                try:
                    inputs[k] = v.to(self.device)
                except Exception:
                    pass

        item['inputs'] = inputs
        item['raw_text_prompt'] = text
        return item
