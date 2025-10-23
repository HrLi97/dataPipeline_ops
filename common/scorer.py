# common/scorer.py
"""
该模块提供用于计算分数的函数。
"""
import numpy as np
from typing import List

def combined_score_from_quality_and_sim(qualities: List[float], sims: List[float]) -> float:
    """
    根据质量分数列表和相似度分数列表计算综合分数。

    综合分数为两组分数平均值的算术平均值。

    参数:
        qualities (List[float]): 质量分数的列表。
        sims (List[float]): 相似度分数的列表。

    返回:
        float: 计算出的综合分数。
    """
    # 如果列表为空，则其平均值视为0.0
    q = float(np.mean(qualities)) if qualities else 0.0
    s = float(np.mean(sims)) if sims else 0.0
    return (q + s) / 2.0