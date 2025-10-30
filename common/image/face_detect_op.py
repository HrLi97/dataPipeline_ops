"""
封装 RetinaFace 人脸检测：输入 frame(RGB)或 crop(BGR),返回 faces dict。
"""
from ..base_ops import BaseOps
from retinaface import RetinaFace 

class FaceDetectOp(BaseOps):
    def __init__(self, **kwargs):
        pass

    def predict(self, item: dict) -> dict:
        # 支持两种用法：若传入 "crop" 则在 crop 上检测；否则在 item["frame"] 全图检测并返回所有 faces
        crop = item.get("crop")  # BGR expected for RetinaFace.detect_faces in your code
        frame = item.get("frame")  # RGB expected -> convert before calling if needed
        try:
            if crop is not None:
                faces = RetinaFace.detect_faces(crop)
                item["faces_in_crop"] = faces
            elif frame is not None:
                import cv2
                bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                faces = RetinaFace.detect_faces(bgr)
                item["faces_in_frame"] = faces
            else:
                item["faces_in_frame"] = {}
        except Exception as e:
            item["face_detect_error"] = str(e)
            item.setdefault("faces_in_frame", {})
        return item
