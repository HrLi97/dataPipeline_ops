"""
加载 Qwen Processor + Model 的算子(负责一次性加载并把实例放到 item 中)。
在多进程/分布式场景下,建议在子进程中构造该算子实例(caption.py 的 Worker 会 lazy init)。
"""
from ..base_ops import BaseOps
from transformers import AutoProcessor, logging as hf_logging
from transformers import Qwen2_5_VLForConditionalGeneration
import torch

hf_logging.set_verbosity_error()

class ModelLoaderOp(BaseOps):
    def __init__(self, model_id: str, min_pixels: int = 16 * 28 * 28, max_pixels: int = 20480 * 28 * 28, dtype=torch.bfloat16, device_map="auto", **kwargs):
        """
        model_id: 本地或 HF 模型路径(比如 /home/xxx/Qwen2.5-VL-32B-Instruct)
        其余参数按需调整
        """
        self.model_id = model_id
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.dtype = dtype
        self.device_map = device_map
        self.processor = None
        self.model = None
        self._loaded = False

    def predict(self, item: dict) -> dict:
        """
        如果尚未加载则加载 processor 与 model;然后把它们写入 item['processor'] 和 item['model']
        """
        if not self._loaded:
            # 防止重复加载(在 lazy-init 场景下 worker 每个进程只加载一次)
            self.processor = AutoProcessor.from_pretrained(self.model_id, min_pixels=self.min_pixels, max_pixels=self.max_pixels)
            # 加载模型(可能消耗显存),保持与原代码相同的参数
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_id, torch_dtype=self.dtype, device_map=self.device_map, attn_implementation="flash_attention_2"
            )
            self._loaded = True

        item['processor'] = self.processor
        item['model'] = self.model
        return item
