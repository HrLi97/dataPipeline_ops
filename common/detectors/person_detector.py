# common/detectors/person_detector.py
"""
该模块提供了对 MMDetection 模型的人物检测功能的封装。
"""
from typing import List, Tuple
import numpy as np

# 在模块顶部提前导入所需的检测函数和模型接口
from mmdet.apis import inference_detector
from utils.find_person_mmdet_one_and_two import detect_person_mmdet

def detect_person_mmdet_wrapper(frame: np.ndarray, mmdet_model, score_thr: float = 0.7) -> List[Tuple[int, int, int, int]]:
    """
    对项目中的 `detect_person_mmdet` 函数进行的一层简单封装。

    此函数直接调用现有的检测逻辑，以保持接口一致性。

    参数:
        frame (np.ndarray): 输入的视频帧 (通常为 BGR 格式)。
        mmdet_model: 加载好的 MMDetection 模型。
        score_thr (float): 用于过滤检测结果的置信度阈值。

    返回:
        List[Tuple[int, int, int, int]]: 检测到的人物边界框列表，格式为 [(x1, y1, x2, y2), ...]。
    """
    # 调用实际的人物检测函数并返回结果
    return detect_person_mmdet(frame, inference_detector, mmdet_model, score_thr=score_thr)