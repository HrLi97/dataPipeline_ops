# common/detectors/face_detector.py
"""
该模块封装了 RetinaFace 人脸检测功能。
"""
from typing import Dict
import numpy as np
from retinaface import RetinaFace

def detect_faces_in_crop(crop_bgr: np.ndarray) -> Dict:
    """
    在指定的 BGR 图像区域中检测人脸。

    参数:
        crop_bgr (np.ndarray): 输入的 BGR 格式图像区域。

    返回:
        Dict: RetinaFace.detect_faces 方法返回的原始结果字典。
    """
    faces = RetinaFace.detect_faces(crop_bgr)
    return faces