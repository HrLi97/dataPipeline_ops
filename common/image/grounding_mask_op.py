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

        try:
            full_rgb = full_bgr[:, :, ::-1].copy() 
            self.image_predictor.set_image(full_rgb)
            import numpy as np
            input_box = np.array([box], dtype=np.float32)
            
            masks, scores, logits = self.image_predictor.predict(
                point_coords=None, point_labels=None, box=input_box, multimask_output=False
            )
            
            mask_full = masks[0].astype(np.uint8) * 255
            item["mask_full"] = mask_full
            return item
            
        except Exception as e:
            import traceback
            print("!!! GroundingMaskOp 发生异常 !!!")
            traceback.print_exc()
            print(f"错误详情: {e}")
            item["mask_full"] = None
            return item

if __name__ == "__main__":
    import os
    import sys
    import cv2
    from hydra import initialize, compose
    from hydra.core.global_hydra import GlobalHydra 
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    cfg_dir = "/datas/workspace/wangshunyao/dataPipeline_ops/third_part/sam2_opt/sam2/sam2"

    current_work_dir = os.getcwd()

    cfg_dir_rel = os.path.relpath(cfg_dir, current_work_dir)

    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    initialize(config_path=cfg_dir_rel, job_name="sam2_test", version_base=None)

    MODEL_CFG = "configs/sam2.1/sam2.1_hiera_l.yaml" 
    SAM2_CHECKPOINT = "/datas/workspace/wangshunyao/dataPipeline_ops/third_part/Grounded_SAM2_opt/checkpoints/sam2.1_hiera_large.pt"
    sam2_image_model = build_sam2(MODEL_CFG, SAM2_CHECKPOINT)
    image_predictor_instance = SAM2ImagePredictor(sam2_image_model)
    print("SAM2 模型加载完毕。")
    
    csv_file_path = "/datas/workspace/wangshunyao/dataPipeline_ops/tmp/image_list.csv"
    with open(csv_file_path, 'r', encoding='utf-8') as f:
        image_path = f.readline().strip()

    bgr_image = cv2.imread(image_path)

    h, w, _ = bgr_image.shape
    test_box = [int(w*0.25), int(h*0.25), int(w*0.75), int(h*0.75)] 

    mask_op = GroundingMaskOp(image_predictor=image_predictor_instance)
    
    test_item = {
        "full_bgr": bgr_image,
        "box": test_box
    }
    result_item = mask_op.predict(test_item)

    if "full_bgr" in result_item:
        del result_item["full_bgr"]

    if "mask_full" in result_item and result_item["mask_full"] is not None:
        mask = result_item.pop("mask_full")
        print("成功生成 Mask,形状为:", mask.shape, "数据类型为:", mask.dtype)
        # (可选) 保存 mask 图像以供查看
        cv2.imwrite("/datas/workspace/wangshunyao/dataPipeline_ops/tmp/mask_output.png", mask)
        print("Mask 图像已保存到 /datas/workspace/wangshunyao/dataPipeline_ops/tmp/mask_output.png")
    else:
        print("生成 Mask 失败。")

    print("最终 item 内容:", result_item)