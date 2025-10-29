# ops/similarity_op.py
"""
封装 AdaFaceMatcher：提供 predict_similarity(a,b) 接口
"""
from ..base_ops import BaseOps

class SimilarityOp(BaseOps):
    def __init__(self, matcher=None, **kwargs):
        # 传入已构造的 AdaFaceMatcher 实例（在 Worker 中创建）
        self.matcher = matcher

    def predict(self, item: dict) -> dict:
        a = item.get("img_a")
        b = item.get("img_b")
        if a is None or b is None or self.matcher is None:
            item["similarity"] = 0.0
            return item
        try:
            sim = float(self.matcher.predict_similarity(a, b))
            item["similarity"] = sim
        except Exception as e:
            item["similarity"] = 0.0
            item["similarity_error"] = str(e)
        return item
