"""
封装 FaceQualityModel:接收 face_img(BGR or RGB),返回 quality score 与 message。
"""
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ..base_ops import BaseOps
from dataPipeline_ops.third_part.CLIB_TensorRT.CLIB_FIQA.inference import FaceQualityModel

class FaceQualityOp(BaseOps):
    def __init__(self, model_path=None, clib_path=None, device="cuda", **kwargs):
        self.model = FaceQualityModel(model_path, clib_path) if model_path else None

    def predict(self, item: dict) -> dict:
        face = item.get("face_img")
        if face is None or self.model is None:
            item["face_quality"] = (0.0, "no_face_or_model")
            return item
        try:
            score, msg = self.model.predict_score(face)
            item["face_quality"] = (float(score), msg)
        except Exception as e:
            item["face_quality"] = (0.0, str(e))
        return item
