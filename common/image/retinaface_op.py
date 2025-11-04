# common/image/retinaface_op.py
"""
RetinaFace 封装。predict 接收 item，期望 item['crop_bgr'] 或 item['crop_rgb']（单张人区域）。
返回 item['faces_in_crop'] = dict (与 RetinaFace.detect_faces 同格式)
"""
from ..base_ops import BaseOps

class RetinaFaceOp(BaseOps):
    def __init__(self):
        # RetinaFace 模块为全局导入使用（无需传 model）
        pass

    def predict(self, item: dict) -> dict:
        crop = item.get("crop_bgr") or item.get("crop_rgb")
        if crop is None:
            item["faces_in_crop"] = {}
            return item
        try:
            # RetinaFace 接收 RGB，需要注意
            import cv2
            from retinaface import RetinaFace
            # 如果 crop 是 BGR -> 转 RGB 输出更稳妥
            # 假定 caller 保证传入 BGR（我们尝试转换为 RGB）
            import numpy as np
            arr = crop
            if arr is None:
                item["faces_in_crop"] = {}
                return item
            # 尝试以 RGB 形式送入 detect_faces（你的代码里用 crop_rgb）
            faces = RetinaFace.detect_faces(arr)
            item["faces_in_crop"] = faces or {}
            return item
        except Exception as e:
            item["faces_in_crop"] = {}
            return item
