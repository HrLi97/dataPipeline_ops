# common/sampler.py
"""
该模块提供用于从视频帧序列中采样索引的函数。
"""
from typing import List, Tuple
import math


def sample_indices(
    total_frames: int, 
    fps: float, 
    ref_interval_sec: float = 5.0, 
    ref_from_last_sec: float = 1.0
) -> Tuple[List[int], List[int]]:
    """
    从视频帧中采样参考帧索引 (ref) 和候选帧索引 (candidate)。

    参考帧从视频尾部的特定时间窗口内采样，候选帧从视频主体部分（后90%）以相同步长采样。

    参数:
        total_frames (int): 视频的总帧数。
        fps (float): 视频的帧率。
        ref_interval_sec (float): 参考帧的采样间隔时间（秒）。
        ref_from_last_sec (float): 用于采样参考帧的末尾时间窗口的长度（秒）。

    返回:
        一个元组，包含两个列表：参考帧索引列表和候选帧索引列表。
    """
    step = max(1, int(ref_interval_sec * fps))
    last_start = max(0, total_frames - int(ref_from_last_sec * fps))
    refs = list(range(last_start, total_frames, step))
    start = total_frames // 10
    cand_step = step
    candidates = list(range(start, total_frames, cand_step))
    return refs, candidates