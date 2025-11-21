"""
BEN2 背景抠除封装。构造时注入 bg_rm_model(BEN2.BEN_Base().to(...).eval())。
predict 期望 item['pil_image'] (PIL) 或 item['face_pil']，返回 item['bg_rm_pil']
"""
from ..base_ops import BaseOps
import torch

class BgRmOp(BaseOps):
    def __init__(self, bg_rm_model=None):
        self.model = bg_rm_model

    def predict(self, item: dict) -> dict:
        pil = item.get("pil_image")
        if pil is None or self.model is None:
            item["bg_rm_pil"] = None
            return item
        try:
            with torch.no_grad():
                out = self.model.inference(pil)
            item["bg_rm_pil"] = out
            return item
        except Exception:
            item["bg_rm_pil"] = None
            return item
