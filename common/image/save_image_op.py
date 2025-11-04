"""
保存图片算子：把传入的各种 np.ndarray / PIL 存到磁盘。
构造时不需要模型, predict 接收 item 并根据 item['save_paths'] 字典保存对应内容。
"""
from ..base_ops import BaseOps
import os
import cv2
import numpy as np
from PIL import Image

def _save_any(path, arr):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if arr is None:
        return
    if isinstance(arr, Image.Image):
        arr.save(path)
        return
    a = arr
    if isinstance(a, np.ndarray):
        if a.dtype in (np.float32, np.float64):
            a = (a * 255).clip(0,255).astype(np.uint8)
        if a.ndim == 3 and a.shape[2] == 3:
            # assume RGB -> convert to BGR for cv2
            # determine whether array is RGB or BGR by user convention; default treat as RGB
            a_bgr = cv2.cvtColor(a, cv2.COLOR_RGB2BGR)
            cv2.imwrite(path, a_bgr)
            return
        if a.ndim == 2:
            cv2.imwrite(path, a)
            return
        # fallback: save with PIL
        Image.fromarray(a).save(path)
        return
    # fallback: try to use cv2.imwrite if arr-like
    try:
        cv2.imwrite(path, arr)
    except Exception:
        pass

class SaveImageOp(BaseOps):
    def __init__(self):
        pass

    def predict(self, item: dict) -> dict:
        save_paths = item.get("save_paths", {})
        # save_paths: dict name->(path, array)
        for k, v in save_paths.items():
            path, arr = v
            try:
                _save_any(path, arr)
            except Exception:
                pass
        item["saved_paths"] = list(save_paths.keys())
        return item
