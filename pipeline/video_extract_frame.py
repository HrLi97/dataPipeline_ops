"""
这是从视频中去获得有效的训练数据对,包括图片完整的描述,和persons>2的mask图片,和原图
首先检查视频的分辨率,分段(5s)提取一张图片,为了防止复制粘贴,从最后的1s提取ref imgs 然后要校验人脸是否一致
"""

import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import csv
import logging
import argparse
import ray
from typing import Optional
import numpy as np
import cv2
from collections import defaultdict

from common.video.video_probe_op import VideoProbeOp
from common.video.decord_reader_op import DecordReaderOp
from common.image.person_detect_op import PersonDetectOp
from common.image.face_detect_op import FaceDetectOp
from common.image.face_quality_op import FaceQualityOp
from common.transform.similarity_op import SimilarityOp
from common.io.save_pairs_op import SavePairsOp

parser = argparse.ArgumentParser(description='scene change detection and transitions detection')
parser.add_argument('--log_path', default='/home/ubuntu/MyFiles/haoran/code/Data_process_talking/pipeline_for_i2i_person/test/log/log.log', help="日志地址")
parser.add_argument('--jsonl_path', default='/home/ubuntu/MyFiles/haoran/code/Data_process_talking/pipeline_for_i2i_person/data/r2v_batch1_out-10min-3.jsonl')
parser.add_argument('--output_dir_root', default='/home/ubuntu/MyFiles/haoran/code/Data_process_talking/pipeline_for_i2i_person/data/r2v_batch1-10min-3/')
parser.add_argument('--input_csv_path', default='/home/ubuntu/MyFiles/haoran/code/Data_process_talking/pipeline_for_i2i_person/data/r2v_batch1_cut-first_10_frames_index.csv')
parser.add_argument('--is_local', default=False, type=bool)
parser.add_argument("--ray_log_dir", type=str, default="/home/ubuntu/MyFiles/haoran/code/Data_process_talking/pipeline_for_i2i_person/test/log")
parser.add_argument("--det_checkpoint", default="/home/ubuntu/MyFiles/haoran/code/Data_process_talking/mmdetection-main/configs/rtmdet/rtmdet_x_8xb32-300e_coco_20220715_230555-cc79b9ae.pth", help="Checkpoint file for detection")
parser.add_argument("--det_config", default="/home/ubuntu/MyFiles/haoran/code/Data_process_talking/mmdetection-main/configs/rtmdet/rtmdet_x_8xb32-300e_coco.py", help="Config file for detection")

parser.add_argument("--min_duration_fps", default=30.0, type=float)
parser.add_argument("--first_frac", default=0.1, type=float)
parser.add_argument("--min_face_score", default=0.8, type=float)
parser.add_argument("--face_quality_thresh", default=0.35, type=float)
parser.add_argument("--face_match_thresh", default=0.5, type=float)
parser.add_argument("--scan_step", default=50, type=int)
parser.add_argument("--sim_lo", default=0.6, type=float)
parser.add_argument("--sim_hi", default=0.85, type=float)
parser.add_argument("--bin_size", default=150, type=int)
parser.add_argument("--min_refs", default=2, type=int)
parser.add_argument("--max_refs", default=3, type=int)

opt = parser.parse_args()


# ----------------- 在 worker/子进程中初始化模型的函数(保持你原来的实现风格) -----------------
def init_models_in_worker(opt):
    """
    在 worker/子进程中初始化重量级模型,并返回实例字典。
    """
    from mmdet.apis import init_detector
    from dataPipeline_ops.third_part.CLIB_TensorRT.CLIB_FIQA.inference import FaceQualityModel
    from dataPipeline_ops.data.a6000_yuan.face_similarity import AdaFaceMatcher

    mmdet_model = init_detector(opt.det_config, opt.det_checkpoint, device="cuda")
    face_quality_model = FaceQualityModel(
        "/home/ubuntu/MyFiles/haoran/code/Data_process_talking/pipeline_for_i2i_person/utile/RN50.pt",
        "/home/ubuntu/MyFiles/haoran/code/Data_process_talking/pipeline_for_i2i_person/utile/CLIB-FIQA_R50.pth"
    )
    adaface = AdaFaceMatcher()

    grounding_model = None
    image_predictor = None
    try:
        # 如果需要加载 SAM / groundingdino 等可以在这里初始化；保持为安全加载
        from sam2.build_sam import build_sam2, build_sam2_video_predictor
        image_predictor = build_sam2()
        # grounding_model = load_grounding_model_safe()
    except Exception:
        grounding_model = None
        image_predictor = None

    return {
        "mmdet_model": mmdet_model,
        "face_quality_model": face_quality_model,
        "adaface": adaface,
        "grounding_model": grounding_model,
        "image_predictor": image_predictor
    }


# ----------------- Worker 类 -----------------
class Worker:
    """
    可调用对象,用于 samples.map(Worker, ...) 或本地串行调用。
    在首次 __call__ 时 lazy init 模型(确保在子进程内初始化)。
    """

    def __init__(self):
        self._initialized = False
        self._models = None

        # 算子实例
        self.probe = None
        self.reader = None
        self.person_detect = None
        self.face_detect = None
        self.face_quality_op = None
        self.sim_op = None
        self.saver = None

    # ---- helper: iou ----
    @staticmethod
    def _compute_iou(boxA, boxB):
        xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
        inter_w = max(0, xB - xA); inter_h = max(0, yB - yA)
        inter = inter_w * inter_h
        areaA = max(0, (boxA[2]-boxA[0]))*max(0, (boxA[3]-boxA[1]))
        areaB = max(0, (boxB[2]-boxB[0]))*max(0, (boxB[3]-boxB[1]))
        union = areaA + areaB - inter
        return inter/union if union>0 else 0.0

    def _lazy_init(self):
        if self._initialized:
            return
        logging.info("Worker lazy init: 初始化模型与算子(子进程/主进程内首次调用)")
        self._models = init_models_in_worker(opt)

        # 构造算子实例(将模型注入需要的算子)
        self.probe = VideoProbeOp()
        self.reader = DecordReaderOp()
        self.person_detect = PersonDetectOp(mmdet_model=self._models["mmdet_model"])
        self.face_detect = FaceDetectOp()
        self.face_quality_op = FaceQualityOp(model_path=None, clib_path=None)
        try:
            # 有些实现把模型对象直接赋给算子实例可用；根据现有的 FaceQualityOp 实现调整
            self.face_quality_op.model = self._models["face_quality_model"]
        except Exception:
            pass
        self.sim_op = SimilarityOp(matcher=self._models["adaface"])
        self.saver = SavePairsOp(output_dir_root=opt.output_dir_root, jsonl_path=opt.jsonl_path)
        self._initialized = True
        logging.info("Worker lazy init 完成")

    def _extract_refs_from_item(self, item: dict) -> dict:
        """
        - 扫描视频前 first_frac 的帧(step=20)
        - 对每个 person box: 检测脸 -> 质量打分 -> 去重(相似度 + iou)
        返回在 item['best_faces'] 中。
        """
        vr = item.get("vr")
        total_frame = item.get("total_frame", 0)
        fps = item.get("fps", None)
        best_faces = []

        if vr is None or total_frame <= 0:
            item["best_faces"] = []
            return item

        upto = max(1, int(total_frame * opt.first_frac))
        indices = list(range(0, upto, 20))
        for frame_idx in indices:
            try:
                frame = vr[frame_idx].asnumpy()  # RGB ndarray
            except Exception:
                continue

            # 检测人体
            p_item = {"frame": frame}
            p_item = self.person_detect.predict(p_item)
            person_boxes = p_item.get("person_boxes", [])
            if not (1 <= len(person_boxes) <= 2):
                continue

            for idx_box, (x1, y1, x2, y2) in enumerate(person_boxes):
                # 裁 crop (注意 decord 返回 RGB,需要转换为 BGR 给 RetinaFace)
                crop_rgb = frame[y1:y2, x1:x2].copy()
                if crop_rgb.size == 0:
                    continue
                crop_bgr = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR)

                f_item = {"crop": crop_bgr}
                f_item = self.face_detect.predict(f_item)
                faces = f_item.get("faces_in_crop", {})
                # 过滤低分脸
                valid = {k: v for k, v in faces.items() if v.get("score", 0) >= opt.min_face_score}
                if len(valid) != 1:
                    continue
                face_key, face_data = next(iter(valid.items()))
                fx1, fy1, fx2, fy2 = face_data["facial_area"]
                # 边界检查
                if fx2 - fx1 <= 0 or fy2 - fy1 <= 0:
                    continue
                face_img = crop_bgr[fy1:fy2, fx1:fx2]

                # face quality
                fq_item = {"face_img": face_img}
                fq_item = self.face_quality_op.predict(fq_item)
                qscore, qmsg = fq_item.get("face_quality", (0.0, ""))
                if qscore <= opt.face_quality_thresh:
                    continue

                # 去重：和已有 best_faces 比相似度 & iou
                matched = False
                for entry in best_faces:
                    sim_item = {"img_a": entry['face'], "img_b": face_img}
                    sim_item = self.sim_op.predict(sim_item)
                    sim = sim_item.get("similarity", 0.0)
                    iou = self._compute_iou(entry['bbox'], (x1, y1, x2, y2))
                    if sim >= opt.face_match_thresh:
                        matched = True
                        if qscore > entry['score']:
                            entry.update({
                                'face': face_img.copy(),
                                'body': crop_bgr.copy(),
                                'bbox': (x1, y1, x2, y2),
                                'score': qscore,
                                'frame_idx': frame_idx
                            })
                        break
                    if sim < opt.face_match_thresh and iou > 0.2:
                        matched = True
                        break
                if not matched:
                    best_faces.append({
                        'face': face_img.copy(),
                        'body': crop_bgr.copy(),
                        'bbox': (x1, y1, x2, y2),
                        'score': qscore,
                        'frame_idx': frame_idx
                    })

        item['best_faces'] = best_faces
        return item

    def _scan_frames_from_item(self, item: dict) -> dict:
        """
        在后 90% 帧中扫描,寻找满足条件的帧；返回 item['frame_scores'](已下采样每 bin 一个最佳)
        """
        vr = item.get("vr")
        total_frame = item.get("total_frame", 0)
        best_faces = item.get("best_faces", [])
        fps = item.get("fps", 25.0)
        if not vr or not best_faces or total_frame <= 0:
            item['frame_scores'] = []
            return item

        split_idx = max(1, total_frame // 10)
        frame_scores = []
        # 从 split_idx 到末尾以 step 步长扫描
        for frame_idx in range(split_idx, total_frame, opt.scan_step):
            try:
                frame = vr[frame_idx].asnumpy()
            except Exception:
                continue

            p_item = {"frame": frame}
            p_item = self.person_detect.predict(p_item)
            person_boxes = p_item.get("person_boxes", [])
            if not (opt.min_refs <= len(person_boxes) <= opt.max_refs):  # reuse min_refs/max_refs for people count
                continue

            all_ok = True
            person_quality = []
            person_sim = []
            matched_refs = []
            for (x1, y1, x2, y2) in person_boxes:
                # 裁取单人区域
                single = frame[y1:y2, x1:x2]
                if single.size == 0:
                    all_ok = False
                    break
                single_bgr = cv2.cvtColor(single, cv2.COLOR_RGB2BGR)
                f_item = {"crop": single_bgr}
                f_item = self.face_detect.predict(f_item)
                faces = f_item.get("faces_in_crop", {})
                valid = {k: v for k, v in faces.items() if v.get("score", 0) >= 0.9}
                if len(valid) < 1:
                    all_ok = False
                    break
                _, face_data = next(iter(valid.items()))
                fx1, fy1, fx2, fy2 = face_data["facial_area"]
                if fx2 - fx1 <= 0 or fy2 - fy1 <= 0:
                    all_ok = False
                    break
                single_face_img = single_bgr[fy1:fy2, fx1:fx2]
                fq_item = {"face_img": single_face_img}
                fq_item = self.face_quality_op.predict(fq_item)
                q, _ = fq_item.get("face_quality", (0.0, ""))
                if q <= opt.face_quality_thresh:
                    all_ok = False
                    break
                person_quality.append(q)
                # 相似度对比
                sims = []
                for entry in best_faces:
                    sim_item = {"img_a": entry['face'], "img_b": single_face_img}
                    sim_item = self.sim_op.predict(sim_item)
                    sims.append(sim_item.get("similarity", 0.0))
                max_sim = max(sims) if sims else 0.0
                if not (opt.sim_lo <= max_sim <= opt.sim_hi):
                    all_ok = False
                    break
                person_sim.append(max_sim)
                matched_refs.append(int(np.argmax(sims)) if sims else -1)
            if not all_ok:
                continue
            avg_q = float(np.mean(person_quality))
            avg_sim = float(np.mean(person_sim))
            combined = (avg_q + avg_sim) / 2.0
            frame_scores.append({'frame_idx': frame_idx, 'combined_score': combined, 'matched_refs': matched_refs})

        # 下采样：分 bin 取每组最高
        groups = defaultdict(list)
        for entry in frame_scores:
            bin_id = entry['frame_idx'] // opt.bin_size
            groups[bin_id].append(entry)
        selected = []
        for bin_id, entries in groups.items():
            best = max(entries, key=lambda x: x['combined_score'])
            selected.append(best)
        selected.sort(key=lambda x: x['frame_idx'])
        item['frame_scores'] = selected
        return item

    # ---- 处理单个视频的主要流程 ----
    def process_single_video(self, vid_path: str) -> bool:
        try:
            item = {"file_path": vid_path}
            item = self.probe.predict(item)
            total_duration = item.get("total_duration", 0.0)

            if total_duration <= 0 or total_duration < opt.min_duration_fps:
                logging.info(f"[skip] video too short or probe failed: {vid_path}, duration={total_duration}")
                return False

            item = self.reader.predict(item)
            if item.get("reader_error"):
                logging.warning(f"[reader error] {vid_path}: {item.get('reader_error')}")
                return False

            # 1) 从前 first_frac 抽取 best_faces(内联)
            item = self._extract_refs_from_item(item)
            best_faces = item.get("best_faces", [])
            if not (opt.min_refs <= len(best_faces) <= opt.max_refs):
                logging.info(f"[skip] ref count not in range for {vid_path}: found {len(best_faces)}")
                return False

            # 2) 在后续帧扫描 candidate frames(内联)
            item = self._scan_frames_from_item(item)
            frame_scores = item.get("frame_scores", [])
            if not frame_scores:
                logging.info(f"[skip] no valid candidate frames for {vid_path}")
                return False

            # 3) 保存 pairs -> jsonl
            item = self.saver.predict(item)
            logging.info(f"[saved] {vid_path} -> {item.get('saved_out_dir')}")
            return True

        except Exception as e:
            logging.exception(f"[error] process_single_video failed for {vid_path}: {e}")
            return False

    # ---- __call__ 保持不变 ----
    def __call__(self, item):
        if not self._initialized:
            self._lazy_init()

        if opt.is_local:
            vid_path = item.get('file_path')
        else:
            vid_path = item.get('file_path') if isinstance(item.get('file_path'), str) else item.get('file_path')[0]

        try:
            logging.info(f"Worker processing {vid_path}")
            self.process_single_video(vid_path)
        except Exception as e:
            logging.exception(f"Worker exception for {vid_path}: {e}")

        return item


# ----------------- main 部分(保持你的原结构) -----------------
if __name__ == '__main__':
    os.makedirs(os.path.dirname(opt.log_path), exist_ok=True, mode=0o777)
    os.makedirs(os.path.dirname(opt.jsonl_path), exist_ok=True, mode=0o777)
    os.makedirs(opt.output_dir_root, exist_ok=True)
    logging.basicConfig(
        filename=opt.log_path,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    if opt.is_local:
        samples = list(csv.DictReader(open(opt.input_csv_path, "r", encoding="utf-8-sig")))
        print(f"本地模式,共读取 {len(samples)} 条记录")
        worker = Worker()
        for item in samples:
            worker(item)
    else:
        ray.init(
            address="auto",
            runtime_env={
                "env_vars": {
                    "PYTHONPATH": ":".join([
                        "/home/ubuntu/Grounded-SAM-2",
                        "/home/ubuntu/MyFiles/haoran/code/Data_process_talking",
                        "/home/ubuntu/MyFiles/haoran/code/Data_process_talking/CLIB-FIQA-main",
                    ]),
                    "HF_ENDPOINT": "https://hf-mirror.com",
                    "DEEPFACE_HOME": "/home/ubuntu/MyFiles/haoran/cpk/",
                }
            }
        )

        samples = ray.data.read_csv(opt.input_csv_path)
        predictions = samples.map(
            Worker,
            num_gpus=1,
            concurrency=19,
        )
        predictions.write_csv(opt.ray_log_dir)

    logging.info("主进程退出")
