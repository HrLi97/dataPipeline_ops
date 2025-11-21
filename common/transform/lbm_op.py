"""
封装 LBM evaluate:构造时注入 lbm_model(get_model 返回的模型)。
predict 期望 item['pil_image'] (PIL.Image)，返回 item['lbm_result_np'] (RGB ndarray)
"""
from ..base_ops import BaseOps
import numpy as np
import torch
import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

class LBMOp(BaseOps):
    def __init__(self, lbm_model=None):
        self.lbm_model = lbm_model

    def predict(self, item: dict) -> dict:
        pil_img = item.get("pil_image")
        if pil_img is None or self.lbm_model is None:
            item["lbm_result_np"] = None
            return item
        try:
            # evaluate(lbm_model, pil_img, num_sampling_steps=1)
            from dataPipeline_ops.third_part.lbm.src.lbm.inference import evaluate
            with torch.no_grad():
                result_pil = evaluate(self.lbm_model, pil_img, num_sampling_steps=1)
            item["lbm_result_np"] = np.array(result_pil)  # RGB
            return item
        except Exception:
            item["lbm_result_np"] = None
            return item
