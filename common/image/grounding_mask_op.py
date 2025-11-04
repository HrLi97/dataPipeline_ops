"""
基于 SAM/Grounding 得到 bbox 对应的 mask)原脚本 get_grounded_mask)。
构造时注入 image_predictor)SAM2ImagePredictor 实例)。
predict 期望 item 包含 'full_bgr' (BGR ndarray) 和 'box' [x1,y1,x2,y2]
返回 item['mask_full'] (单通道 0/255 uint8)
"""
from ..base_ops import BaseOps
import numpy as np

class GroundingMaskOp(BaseOps):
    def __init__(self, image_predictor=None):
        self.image_predictor = image_predictor

    def predict(self, item: dict) -> dict:
        full_bgr = item.get("full_bgr")
        box = item.get("box")
        if full_bgr is None or box is None or self.image_predictor is None:
            item["mask_full"] = None
            return item
        # 使用 image_predictor 得到 mask)基于原脚本)
        try:
            full_rgb = full_bgr[:, :, ::-1]  # BGR -> RGB
            self.image_predictor.set_image(full_rgb)
            import numpy as np
            input_box = np.array([box], dtype=np.float32)
            masks, scores, logits = self.image_predictor.predict(
                point_coords=None, point_labels=None, box=input_box, multimask_output=False
            )
            mask_full = masks[0].astype(np.uint8) * 255
            item["mask_full"] = mask_full
            return item
        except Exception:
            item["mask_full"] = None
            return item
