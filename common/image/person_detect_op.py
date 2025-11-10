"""
封装 mmdet/person 检测
改进点：
- 保留并把原始导入放到文件顶部(如你要求)
- 支持 item['frame'](RGB)或 item['image'](BGR)
- 若首次检测失败,会尝试把通道顺序转换后重试
- 规范输出 item['person_boxes'] 为 list of (int,int,int,int)
- 出错信息写入 item['person_detect_error']
"""
import sys
import os
import logging
import cv2
import numpy as np
from mmdet.apis import inference_detector
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from dataPipeline_ops.data.a6000_yuan.find_person_mmdet_one_and_two import detect_person_mmdet

from ..base_ops import BaseOps

detect_person_mmdet = detect_person_mmdet

class PersonDetectOp(BaseOps):
    def __init__(self, mmdet_model=None, score_thr=0.7, **kwargs):
        """
        Args:
            mmdet_model: 已初始化的 mmdet 模型实例(在 Worker 中注入)
            score_thr: 检测置信度阈值(传给 detect_person_mmdet)
        """
        self.mmdet_model = mmdet_model
        self.score_thr = score_thr

    def _normalize_boxes(self, boxes):
        """把 boxes 转成 list of (int,int,int,int)"""
        norm = []
        if not boxes:
            return norm
        for b in boxes:
            try:
                # 支持常见的 list/tuple/ndarray 行为
                x1, y1, x2, y2 = b
                norm.append((int(x1), int(y1), int(x2), int(y2)))
            except Exception:
                # 兜底解析：若是更复杂结构尝试常见索引
                try:
                    # 例如 [ [x1,y1], [x2,y2] ]
                    x1 = int(b[0][0]); y1 = int(b[0][1]); x2 = int(b[1][0]); y2 = int(b[1][1])
                    norm.append((x1, y1, x2, y2))
                except Exception:
                    continue
        return norm

    def predict(self, item: dict) -> dict:
        """
        item 输入：
            frame: RGB ndarray(优先)
            或 image: BGR ndarray
        输出：
            item['person_boxes'] -> list of (x1,y1,x2,y2)
            若出错,item['person_detect_error'] = 错误信息
        """
        frame = item.get("frame", None)
        image = item.get("image", None)

        img = None
        used_key = None
        if frame is not None:
            img = frame
            used_key = 'frame'
        elif image is not None:
            img = image
            used_key = 'image'

        if img is None:
            item["person_boxes"] = []
            return item

        # 若 detect_person_mmdet 未能导入,直接返回空并写错误信息
        if detect_person_mmdet is None:
            item["person_detect_error"] = "detect_person_mmdet 未能导入(请检查 dataPipeline_ops 路径或 detect_body 模块)"
            item["person_boxes"] = []
            return item

        boxes = []
        last_err = None

        # 首次尝试调用(以调用方传入的通道顺序为准)
        try:
            boxes = detect_person_mmdet(img, inference_detector, self.mmdet_model, score_thr=self.score_thr) or []
        except Exception as e:
            last_err = e
            logging.debug("person_detect_op: 初次 detect_person_mmdet 抛错: %s", e)
            boxes = []

        # 如果没有检测到框,尝试做 RGB<->BGR 转换再试一次(以排查通道问题)
        if not boxes:
            try:
                # 仅当 img 是三通道时尝试转换
                if isinstance(img, np.ndarray) and img.ndim == 3 and img.shape[2] == 3:
                    if used_key == 'frame':
                        alt = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    else:
                        alt = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    try:
                        boxes = detect_person_mmdet(alt, None, self.mmdet_model, score_thr=self.score_thr) or []
                        if boxes:
                            logging.debug("person_detect_op: 通道互转后检测成功(可能是 RGB/BGR 传入不一致)")
                    except Exception as e2:
                        last_err = e2
                        boxes = []
            except Exception as e_conv:
                logging.debug("person_detect_op: 做通道转换时出错: %s", e_conv)

        # 规范化 boxes 并返回
        try:
            norm_boxes = self._normalize_boxes(boxes)
            item["person_boxes"] = norm_boxes
            if not norm_boxes and last_err is not None:
                item["person_detect_error"] = str(last_err)
        except Exception as e_norm:
            item["person_detect_error"] = f"boxes 解析错误: {e_norm}"
            item["person_boxes"] = []

        return item

if __name__ == "__main__":
    from mmdet.apis import init_detector

    DET_CONFIG = "/datas/workspace/wangshunyao/dataPipeline_ops/third_part/mmdetection-main/configs/rtmdet/rtmdet_x_8xb32-300e_coco.py"
    DET_CHECKPOINT = "/datas/workspace/wangshunyao/dataPipeline_ops/third_part/mmdetection-main/configs/rtmdet/rtmdet_x_8xb32-300e_coco_20220715_230555-cc79b9ae.pth"
    mmdet_model_instance = init_detector(DET_CONFIG, DET_CHECKPOINT, device="cuda")

    csv_file_path = "/datas/workspace/wangshunyao/dataPipeline_ops/tmp/image_list.csv"
    
    with open(csv_file_path, 'r', encoding='utf-8') as f:
        image_path = f.readline().strip()

    # 读取 BGR 图像，并将其放入 'image' 键，以测试算子的 BGR 输入路径
    bgr_image = cv2.imread(image_path)
    
    person_op = PersonDetectOp(mmdet_model=mmdet_model_instance, score_thr=0.5)
    
    test_item = {"image": bgr_image}
    result_item = person_op.predict(test_item)

    # 为了不在终端打印巨大的图像数组，我们先将其从结果中移除
    if "image" in result_item:
        del result_item["image"]
    
    print(result_item)