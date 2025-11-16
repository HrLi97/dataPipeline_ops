"""
RetinaFace 封装。predict 接收 item,期望 item['crop_bgr'] 或 item['crop_rgb']（单张人区域）。
返回 item['faces_in_crop'] = dict (与 RetinaFace.detect_faces 同格式)
"""
from ..base_ops import BaseOps

class RetinaFaceOp(BaseOps):
    def __init__(self):
        # RetinaFace 模块为全局导入使用（无需传 model）
        pass

    def predict(self, item: dict) -> dict:
        crop = item.get("crop_bgr")
        if crop is None:
            crop = item.get("crop_rgb")
        if crop is None:
            item["faces_in_crop"] = {}
            return item
        try:
            # RetinaFace 接收 RGB，需要注意
            import cv2
            from retinaface import RetinaFace
            # 如果 crop 是 BGR -> 转 RGB 输出更稳妥
            # 假定 caller 保证传入 BGR（我们尝试转换为 RGB）
            import numpy as np
            arr = crop
            if arr is None:
                item["faces_in_crop"] = {}
                return item
            
            if arr.dtype != np.uint8:
                arr = (np.clip(arr, 0, 255)).astype(np.uint8)

            # If grayscale -> convert to BGR
            if arr.ndim == 2:
                arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)

            # If caller passed 'crop_bgr' convert to RGB, else assume RGB
            if "crop_bgr" in item:
                rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
            else:
                rgb = arr.copy()

            rgb = np.ascontiguousarray(rgb)
            
            faces = RetinaFace.detect_faces(rgb)
            item["faces_in_crop"] = faces or {}
            return item
        except Exception as e:
            item["faces_in_crop"] = {}
            return item

if __name__ == '__main__':
    import cv2
    import pprint

    csv_file_path = "/datas/workspace/wangshunyao/dataPipeline_ops/tmp/image_list.csv"

    with open(csv_file_path, 'r', encoding='utf-8') as f:
        source_image_path = f.readline().strip()
    
    image_bgr = cv2.imread(source_image_path)

    op = RetinaFaceOp()
    item = {"crop_bgr": image_bgr}
    result_item = op.predict(item)

    print("\nDetection Result (item['faces_in_crop']):")
    faces = result_item.get("faces_in_crop", {})
    pprint.pprint(faces)