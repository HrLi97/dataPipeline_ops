# common/validators.py
"""
该模块提供了一系列用于验证和过滤检测结果的函数。
"""
import numpy as np
import cv2
from typing import Tuple, List
from retinaface import RetinaFace


def detect_top_bottom(frame: np.ndarray, black_threshold: int = 10) -> Tuple[int, int]:
    """
    检测视频帧顶部和底部的黑边，返回有效图像区域的垂直范围。

    参数:
        frame (np.ndarray): 输入的视频帧。
        black_threshold (int): 判断像素是否为黑色的灰度阈值。

    返回:
        一个元组，包含有效区域的顶部和底部行的索引。
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h = gray.shape[0]
    top = 0
    for i in range(h):
        if not np.all(gray[i] < black_threshold):
            top = i
            break
    bottom = h - 1
    for i in range(h - 1, -1, -1):
        if not np.all(gray[i] < black_threshold):
            bottom = i
            break
    return top, bottom


def filter_person_by_area(person_box: tuple, frame_width: int, frame_height: int, area_ratio_thresh: float=0.3) -> List[tuple]:
    """
    根据人物边界框的面积与总帧面积的比例来过滤人物。

    参数:
        person_box (tuple): 人物边界框坐标 (x1, y1, x2, y2)。
        frame_width (int): 视频帧的宽度。
        frame_height (int): 视频帧的高度。
        area_ratio_thresh (float): 面积比例阈值。

    返回:
        如果满足条件，返回包含该边界框的列表，否则返回空列表。
    """
    frame_area = frame_width * frame_height
    x1, y1, x2, y2 = person_box
    box_area = max(0, x2 - x1) * max(0, y2 - y1)
    if box_area / frame_area >= area_ratio_thresh:
        return [(x1, y1, x2, y2)]
    return []


def compute_iou(boxA, boxB) -> float:
    """
    计算两个边界框之间的交并比 (Intersection over Union, IoU)。

    参数:
        boxA (tuple): 第一个边界框 (x1, y1, x2, y2)。
        boxB (tuple): 第二个边界框 (x1, y1, x2, y2)。

    返回:
        float: IoU值。
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = max(0, (boxA[2]-boxA[0])) * max(0, (boxA[3]-boxA[1]))
    areaB = max(0, (boxB[2]-boxB[0])) * max(0, (boxB[3]-boxB[1]))
    union = areaA + areaB - inter
    return inter/union if union>0 else 0.0


def filter_person_by_face(frame, person_box, iou_area_thresh=0.1) -> Tuple[bool, float, List[int]]:
    """
    通过在人物边界框内检测人脸来过滤人物。

    如果检测到的人脸面积与人物框面积比例满足阈值，则认为有效。

    参数:
        frame (np.ndarray): RGB格式的视频帧。
        person_box (tuple): 人物边界框坐标。
        iou_area_thresh (float): 人脸面积与身体面积的比例阈值。

    返回:
        一个元组 (is_valid, best_score, best_face_box)，分别表示是否有效、
        最佳人脸分数和最佳人脸框坐标。
    """
    x1, y1, x2, y2 = person_box
    body_w = x2 - x1
    body_h = y2 - y1
    if body_w <= 0 or body_h <= 0:
        return False, 0.0, []
    
    body_area = body_w * body_h
    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    H, W = bgr.shape[:2]
    
    # 裁剪坐标以确保在图像边界内
    x1_clamped, y1_clamped = max(0, min(x1, W-1)), max(0, min(y1, H-1))
    x2_clamped, y2_clamped = max(0, min(x2, W)), max(0, min(y2, H))
    if x2_clamped <= x1_clamped or y2_clamped <= y1_clamped:
        return False, 0.0, []

    # 在人物框内检测人脸
    roi_single = bgr[y1_clamped:y2_clamped, x1_clamped:x2_clamped]
    faces = RetinaFace.detect_faces(roi_single)
    if not isinstance(faces, dict) or not faces:
        return False, 0.0, []
        
    # 找到分数最高的人脸并进行验证
    best_key, best_val = max(faces.items(), key=lambda kv: kv[1].get("score", 0.0))
    best_score = best_val.get("score", 0.0)
    fx1_rel, fy1_rel, fx2_rel, fy2_rel = best_val["facial_area"]
    face_area = max(0, fx2_rel - fx1_rel) * max(0, fy2_rel - fy1_rel)
    
    if face_area / float(body_area) < iou_area_thresh:
        return False, 0.0, []
        
    # 计算人脸的绝对坐标
    abs_fx1 = int(fx1_rel + x1_clamped)
    abs_fy1 = int(fy1_rel + y1_clamped)
    abs_fx2 = int(fx2_rel + x1_clamped)
    abs_fy2 = int(fy2_rel + y1_clamped)
    best_face_box = [abs_fx1, abs_fy1, abs_fx2, abs_fy2]
    
    return True, best_score, best_face_box