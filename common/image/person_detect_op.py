"""
封装 mmdet/person 检测（复用你现有的 detect_person_mmdet);
输入 item 包含 'frame' ndarray(RGB),返回 item['person_boxes']（list of boxes）。
"""
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ..base_ops import BaseOps
from dataPipeline_ops.data.a6000_yuan.find_person_mmdet_one_and_two import detect_person_mmdet
import numpy as np

class PersonDetectOp(BaseOps):
    def __init__(self, mmdet_model=None, score_thr=0.7, **kwargs):
        self.mmdet_model = mmdet_model  # 在 Worker 中传入已 init 的模型
        self.score_thr = score_thr

    def predict(self, item: dict) -> dict:
        frame = item.get("frame")  # RGB ndarray
        if frame is None:
            item["person_boxes"] = []
            return item
        try:
            boxes = detect_person_mmdet(frame, None, self.mmdet_model, score_thr=self.score_thr)
            # boxes expected as list of (x1,y1,x2,y2)
            item["person_boxes"] = boxes
        except Exception as e:
            item["person_detect_error"] = str(e)
            item["person_boxes"] = []
        return item
