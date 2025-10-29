"""
封装 FaceQualityModel:接收 face_img(BGR or RGB),返回 quality score 与 message。
"""
from ..base_ops import BaseOps
from inference import FaceQualityModel

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
