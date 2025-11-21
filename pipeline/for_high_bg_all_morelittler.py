"""
Pipeline: for_high_bg_all_morelittler.py
功能：从 jsonl 读入记录 -> 对 input_images 与 output_image 运行人物检测/抠图/背景去除/保存 -> 结果追加写 output_jsonl
"""
import os
import json
import csv
import argparse
import logging
import ray
import cv2
import numpy as np
import torch
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from mmdet.apis import init_detector
from third_part.lbm.src.lbm.inference import get_model
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from third_part.toolsbg.BNE2 import BEN_Base
from groundingdino.util.inference import load_model

# 从 common 导入算子
from common.image.person_detect_op import PersonDetectOp
from common.image.retinaface_op import RetinaFaceOp
from common.image.grounding_mask_op import GroundingMaskOp
from common.transform.lbm_op import LBMOp
from common.transform.bg_rm_op import BgRmOp
from common.image.save_image_op import SaveImageOp
from common.io.save_jsonl_op import SaveJsonlOp

parser = argparse.ArgumentParser(description='high bg pipeline')
parser.add_argument('--input_json_path', default='/home/ubuntu/.../web-image-multipel_and_single.json')
parser.add_argument('--output_jsonl_root', default='/home/ubuntu/.../out.jsonl')
parser.add_argument('--output_dir_root', default='/home/ubuntu/.../out_dir/')
parser.add_argument('--is_local', default=False, type=bool)
parser.add_argument("--ray_log_dir", type=str, default="/home/ubuntu/.../ray_log")
parser.add_argument("--det_checkpoint", default="/datas/workspace/wangshunyao/dataPipeline_ops/third_part/mmdetection-main/configs/rtmdet/rtmdet_x_8xb32-300e_coco_20220715_230555-cc79b9ae.pth", help="Checkpoint file for detection")
parser.add_argument("--det_config", default="/datas/workspace/wangshunyao/dataPipeline_ops/third_part/mmdetection-main/configs/rtmdet/rtmdet_x_8xb32-300e_coco.py")
opt = parser.parse_args()

GROUNDING_DINO_CONFIG = "dataPipeline_ops/third_part/Grounded_SAM2_opt/grounding_dino"
GROUNDING_DINO_CHECKPOINT = "dataPipeline_ops/third_part/Grounded_SAM2_opt/gdino_checkpoints"

# ---- Worker 类（lazy init） ----
class Worker:
    def __init__(self):
        self._inited = False
        # 算子占位
        self.person_detect_op = None
        self.retina_op = None
        self.grounding_op = None
        self.lbm_op = None
        self.bg_rm_op = None
        self.save_image_op = None
        self.save_jsonl_op = None

        # 模型引用（会在 lazy_init 中创建）
        self.mmdet_model = None
        self.image_predictor = None
        self.lbm_model = None
        self.bg_rm_model = None

    def _lazy_init(self):
        if self._inited:
            return
        logging.info("Worker lazy init: 在当前进程中加载模型并构造算子")

        self.mmdet_model = init_detector(opt.det_config, opt.det_checkpoint, device="cuda")
        # grounding + sam
        try:
            gm = load_model(model_config_path=GROUNDING_DINO_CONFIG, model_checkpoint_path=GROUNDING_DINO_CHECKPOINT, device="cuda")
        except Exception:
            gm = None
        self.grounding_model = gm
        sam_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
        sam_ckpt = "/datas/workspace/wangshunyao/dataPipeline_ops/third_part/Grounded_SAM2_opt/checkpoints/sam2.1_hiera_large.pt"
        sam_model = build_sam2(sam_cfg, sam_ckpt)
        self.image_predictor = SAM2ImagePredictor(sam_model)
        self.lbm_model = get_model("/datas/workspace/wangshunyao/dataPipeline_ops/third_part/LBM_relighting", torch_dtype=torch.bfloat16, device="cuda")
        self.bg_rm_model = BEN_Base().to("cuda").eval()
        self.bg_rm_model.loadcheckpoints('/datas/workspace/wangshunyao/dataPipeline_ops/third_part/toolsbg/BEN2_Base.pth')

        # 2) 构造算子并注入模型实例
        self.person_detect_op = PersonDetectOp(mmdet_model=self.mmdet_model, score_thr=0.80)
        self.retina_op = RetinaFaceOp()
        self.grounding_op = GroundingMaskOp(image_predictor=self.image_predictor)
        self.lbm_op = LBMOp(lbm_model=self.lbm_model)
        self.bg_rm_op = BgRmOp(bg_rm_model=self.bg_rm_model)
        self.save_image_op = SaveImageOp()
        self.save_jsonl_op = SaveJsonlOp(output_jsonl_path=opt.output_jsonl_root)

        self._inited = True
        logging.info("Worker lazy init 完成")

    # 处理单条 json_line 的方法（调用上面各个算子）
    def _process_single_json(self, json_line: dict):
        """
        json_line: 最小字段：
          - input_images: list 或 str
          - output_image: str
        处理逻辑尽量复刻原脚本 process_images/process_one
        """
        out_root = opt.output_dir_root
        input_paths = json_line.get("input_images", [])
        if not isinstance(input_paths, (list, tuple)):
            input_paths = [input_paths]
        output_path = json_line.get("output_image")

        any_saved = False
        # 先批量处理 input_images
        inputs_info = []
        for p in input_paths:
            persons, ok = self._process_one(p, tag="in", out_root=out_root)
            if ok:
                inputs_info.append({"path": p, "persons": persons})
                any_saved = True

        # 处理 output image
        out_persons, out_ok = self._process_one(output_path, tag="out", out_root=out_root)
        if out_ok:
            any_saved = True

        json_line["input_persons"] = inputs_info
        json_line["output_persons"] = out_persons

        if any_saved:
            # 写 jsonl
            item = {"json_line": json_line}
            self.save_jsonl_op.predict(item)

    def _process_one(self, image_path: str, tag: str, out_root: str):
        """
        单张图片处理流程（基于原脚本 process_one）：
         - 读取图片 -> 分辨率过滤
         - mmdet 人体检测（1~3 人）
         - 对每个 box: retinaface 检测人脸 (score>=0.5) -> grounding mask -> bg rm -> lbm -> 保存若干文件
        返回: (persons_list, flag_saved_boolean)
        """
        if not image_path or not os.path.exists(image_path):
            return [], False
        img = cv2.imread(image_path)
        if img is None:
            return [], False
        h, w = img.shape[:2]
        if h < 1000 or w < 1000:
            return [], False

        # 1) person detect
        item = {"image": img}
        item = self.person_detect_op.predict(item)
        person_boxes = item.get("person_boxes", [])
        if not (1 <= len(person_boxes) <= 3):
            return [], False

        IMG_AREA = float(h * w)
        max_area = 0.0
        def clipped_area(b):
            x1,y1,x2,y2 = [int(v) for v in b]
            x1 = max(0, min(x1, w)); x2 = max(0, min(x2, w))
            y1 = max(0, min(y1, h)); y2 = max(0, min(y2, h))
            return max(0, x2-x1)*max(0, y2-y1)
        max_box_area = max((clipped_area(b) for b in person_boxes), default=0)
        if IMG_AREA <=0 or (max_box_area/IMG_AREA) > 0.4:
            return [], False

        # create save dir
        fileName = os.path.splitext(os.path.basename(image_path))[0]
        level2 = os.path.basename(os.path.dirname(image_path))
        save_dir = os.path.join(out_root, level2, tag)
        os.makedirs(save_dir, exist_ok=True)
        orig_out = os.path.join(save_dir, f"{fileName}.jpg")
        if os.path.exists(orig_out):
            # 如果已存在则当作已处理
            return [], orig_out

        persons = []
        for idx, box in enumerate(person_boxes):
            x1,y1,x2,y2 = [int(v) for v in box]
            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            # retinaface 检测
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            f_item = {"crop_rgb": crop_rgb}
            f_item = self.retina_op.predict(f_item)
            faces = f_item.get("faces_in_crop", {})
            vf = {k:v for k,v in faces.items() if v.get("score",0) >= 0.5}
            if len(vf) != 1:
                continue
            # grounding mask
            gm_item = {"full_bgr": img, "box": [x1,y1,x2,y2]}
            gm_item = self.grounding_op.predict(gm_item)
            mask_full = gm_item.get("mask_full")
            if mask_full is None:
                continue
            coords = cv2.findNonZero(mask_full)
            if coords is None:
                continue
            xx, yy, ww, hh = cv2.boundingRect(coords)
            mask_bbox = mask_full[yy:yy+hh, xx:xx+ww]
            img_bbox = img[yy:yy+hh, xx:xx+ww]
            # bg only
            inv_mask = cv2.bitwise_not(mask_full)
            bg_only = cv2.bitwise_and(img, img, mask=inv_mask)

            # LBM + BG_RM: prepare PIL for bg_rm and lbm
            from PIL import Image
            pil_obj = Image.fromarray(cv2.cvtColor(img_bbox, cv2.COLOR_BGR2RGB))

            # bg rm (BEN2)
            br_item = {"pil_image": pil_obj}
            br_item = self.bg_rm_op.predict(br_item)
            bg_rm_pil = br_item.get("bg_rm_pil")
            # lbm evaluate
            lbm_item = {"pil_image": pil_obj}
            lbm_item = self.lbm_op.predict(lbm_item)
            lbm_np = lbm_item.get("lbm_result_np")  # RGB

            # compose some outputs
            suffix = f"_{tag}_p{idx}"
            paths = {
                "mask_bbox": (os.path.join(save_dir, f"{fileName}{suffix}_mask_bbox.png"), mask_bbox),
                "human_bbox": (os.path.join(save_dir, f"{fileName}{suffix}_human_bbox.png"), img_bbox),
                "human_lbm": (os.path.join(save_dir, f"{fileName}{suffix}_human_lbm.jpg"), lbm_np[:, :, ::-1] if lbm_np is not None else None), # RGB->BGR
                "background": (os.path.join(save_dir, f"{fileName}{suffix}_background.png"), bg_only),
                "background_mask": (os.path.join(save_dir, f"{fileName}{suffix}_background_mask.png"), mask_full),
                "human_full_bgr": (os.path.join(save_dir, f"{fileName}{suffix}_human_full_bgr.png"), cv2.bitwise_and(img, img, mask=mask_full)),
                # 你可以按需增加更多保存项
            }
            # 保存
            save_item = {"save_paths": paths}
            self.save_image_op.predict(save_item)

            persons.append({
                "box": [x1,y1,x2,y2],
                **{k: v[0] for k, v in paths.items()}
            })
        # 最后保存原图副本
        cv2.imwrite(orig_out, img)
        return persons, True

    def __call__(self, item):
        if not self._inited:
            self._lazy_init()
        try:
            self._process_single_json(item)
        except Exception:
            logging.exception("Worker 处理单条 json 出错")
        return item

if __name__ == '__main__':
    os.makedirs(os.path.dirname(opt.input_json_path), exist_ok=True, mode=0o777)
    os.makedirs(opt.output_dir_root, exist_ok=True)
    logging.basicConfig(level=logging.INFO)

    if opt.is_local:
        samples = []
        with open(opt.input_json_path, "r") as jf:
            for line in jf:
                s = line.strip()
                if not s:
                    continue
                samples.append(json.loads(s))
        worker = Worker()
        for item in samples:
            worker(item)
    else:
        ray.init(
            address="auto",
            runtime_env={
                "env_vars": {
                    "PYTHONPATH": ":".join([
                        "/home/ubuntu/MyFiles/haoran/code/Data_process_talking",
                        "/home/ubuntu/Grounded-SAM-2",
                        # add others as you had
                    ]),
                    "HF_ENDPOINT": "https://hf-mirror.com",
                }
            }
        )
        samples = ray.data.read_json(opt.input_json_path)
        predictions = samples.map(Worker, num_gpus=1, concurrency=6)
        predictions.write_json(opt.ray_log_dir)
