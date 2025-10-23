# pipelines/sam2_valid_frame/run.py
import os
import cv2
import json
import numpy as np
import logging
from collections import defaultdict
import argparse

# common modules (assumed to be in PYTHONPATH or package)
from common.video_io import VideoIO
from common.sampler import sample_indices
from common.io_ops import save_image, append_jsonl
from common.detectors.person_detector import detect_person_mmdet_wrapper
from common.detectors.face_detector import detect_faces_in_crop
from common.heavy_ops.face_matcher import FaceMatcher
from common.validators import detect_top_bottom, filter_person_by_area, compute_iou
from common.scorer import combined_score_from_quality_and_sim

# Optional: ray usage kept similar to original, will only import/use if not local mode
try:
    import ray
except Exception:
    ray = None

from mmdet.apis import init_detector
from inference import FaceQualityModel

class Worker:
    def __init__(self, opt):
        """
        opt: parsed argparse namespace containing model paths and thresholds.
        Keep behavior similar to original script: init FaceQualityModel and mmdet model here.
        """
        self.opt = opt
    
        # initialize face quality model (paths provided via opt)
        self.face_quality = FaceQualityModel(opt.face_quality_ckpt_path, opt.face_quality_other_ckpt)

        # face matcher (AdaFace wrapper lazy-loads inside)
        self.adaface_similarity = FaceMatcher()

        # init_detector may be heavy; we keep it here to mimic original code
        self.mmdet_model = init_detector(opt.det_config, opt.det_checkpoint, device="cuda")

    def process_video(self, video_path):
        """
        Core processing of a single video.
        Steps preserved from original script, but using common/* helpers.
        """
        name = os.path.basename(video_path).split('.')[0]
        root_path = self.opt.output_dir_root
        os.makedirs(root_path, exist_ok=True)
        out_dir = os.path.join(root_path, name)
        if os.path.exists(out_dir):
            logging.info(f"Skip existing output dir: {out_dir}")
            return

        # open video
        try:
            v = VideoIO(video_path, use_gpu=False)
        except Exception as e:
            logging.error(f"Failed to open video {video_path}: {e}")
            return

        meta = v.metadata()
        width, height = meta['width'], meta['height']
        total_frame = meta['total_frames']
        fps = meta['fps']

        # resolution check as original
        if width < 1920 or height < 1080:
            logging.info(f"Video resolution too small: {video_path} ({width}x{height})")
            v.release()
            return

        # sampling indices (mirrors original: refs from last 1s, candidates from last 90% starting at 10%)
        ref_indices, candidate_indices = sample_indices(total_frame, fps, ref_interval_sec=5.0, ref_from_last_sec=1.0)

        # find best ref faces from first 10% region (sample step 20)
        split_idx = total_frame // 10
        best_faces = []  # entries: dict with keys face, body, bbox, score, output_msg, frame_idx, box_idx

        for frame_idx in range(0, split_idx, 20):
            try:
                frame = v.get_frame(frame_idx)  # RGB ndarray
            except Exception as e:
                logging.warning(f"Failed get_frame {frame_idx} for {video_path}: {e}")
                continue

            # detect persons (thin wrapper that calls your detect_person_mmdet)
            person_boxes = detect_person_mmdet_wrapper(frame, self.mmdet_model, score_thr=0.7)
            if not (1 <= len(person_boxes) <= 2):
                continue

            for idx, (x1, y1, x2, y2) in enumerate(person_boxes):
                # crop the person box
                # guard coordinates
                x1c, y1c, x2c, y2c = max(0, x1), max(0, y1), min(frame.shape[1], x2), min(frame.shape[0], y2)
                if x2c <= x1c or y2c <= y1c:
                    continue
                crop = frame[y1c:y2c, x1c:x2c]
                crop_bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)

                # detect faces within the person crop
                try:
                    faces = detect_faces_in_crop(crop_bgr)
                except Exception as e:
                    logging.warning(f"RetinaFace detect failed at frame {frame_idx}: {e}")
                    continue

                # filter by score >= 0.8 (same as original)
                valid_faces = {k: v for k, v in faces.items() if v.get('score', 0) >= 0.8}
                if len(valid_faces) != 1:
                    continue

                face_key, face_data = next(iter(valid_faces.items()))
                x1f, y1f, x2f, y2f = face_data['facial_area']
                # ensure coordinates valid
                if x2f <= x1f or y2f <= y1f:
                    continue
                face_img = crop[y1f:y2f, x1f:x2f]

                # face quality scoring
                try:
                    quality_preds, output_msg = self.face_quality.predict_score(face_img)
                except Exception as e:
                    logging.warning(f"FaceQuality predict failed: {e}")
                    continue

                if quality_preds <= self.opt.face_quality_thresh:
                    continue

                # deduplicate & maintain best by quality as original
                matched = False
                for entry in best_faces:
                    sim = self.adaface_similarity.predict_similarity(entry['face'], face_img)
                    iou = compute_iou(entry['bbox'], (x1c, y1c, x2c, y2c))
                    if sim >= self.opt.face_match_thresh:
                        matched = True
                        if quality_preds > entry['score']:
                            entry.update({
                                'face': face_img.copy(),
                                'body': crop.copy(),
                                'bbox': (x1c, y1c, x2c, y2c),
                                'score': quality_preds,
                                'output_msg': output_msg,
                                'frame_idx': frame_idx,
                                'box_idx': idx
                            })
                        break
                    if sim < self.opt.face_match_thresh and iou > 0.2:
                        matched = True
                        break

                if not matched:
                    best_faces.append({
                        'face': face_img.copy(),
                        'body': crop.copy(),
                        'bbox': (x1c, y1c, x2c, y2c),
                        'score': quality_preds,
                        'output_msg': output_msg,
                        'frame_idx': frame_idx,
                        'box_idx': idx
                    })

        # require 2-3 refs as original
        if not (2 <= len(best_faces) <= 3):
            logging.info(f"Not enough refs found ({len(best_faces)}) in {video_path}")
            v.release()
            return

        # evaluate candidate frames (sample every 50 frames as original)
        frame_scores = []
        for frame_idx in range(split_idx, total_frame, 50):
            try:
                frame = v.get_frame(frame_idx)
            except Exception as e:
                logging.warning(f"Failed get_frame {frame_idx}: {e}")
                continue

            all_ok = True
            person_quality = []
            person_sim = []
            matched_refs = []

            person_boxes = detect_person_mmdet_wrapper(frame, self.mmdet_model, score_thr=0.7)
            if not (2 <= len(person_boxes) <= 3):
                continue

            for (x1, y1, x2, y2) in person_boxes:
                x1c, y1c, x2c, y2c = max(0, x1), max(0, y1), min(frame.shape[1], x2), min(frame.shape[0], y2)
                if x2c <= x1c or y2c <= y1c:
                    all_ok = False
                    break

                single_person = frame[y1c:y2c, x1c:x2c]
                single_person_bgr = cv2.cvtColor(single_person, cv2.COLOR_RGB2BGR)

                # detect faces in this person crop
                try:
                    faces = detect_faces_in_crop(single_person_bgr)
                except Exception as e:
                    logging.warning(f"Face detect failed at frame {frame_idx}: {e}")
                    all_ok = False
                    break

                valid_faces = {k: v for k, v in faces.items() if v.get('score', 0) >= 0.9}
                if len(valid_faces) < 1:
                    all_ok = False
                    break

                face_key, face_data = next(iter(valid_faces.items()))
                x1f, y1f, x2f, y2f = face_data['facial_area']
                if x2f <= x1f or y2f <= y1f:
                    all_ok = False
                    break
                single_face_img = single_person_bgr[y1f:y2f, x1f:x2f]

                try:
                    quality_preds, output_msg = self.face_quality.predict_score(single_face_img)
                except Exception as e:
                    logging.warning(f"FaceQuality predict failed: {e}")
                    all_ok = False
                    break

                if quality_preds <= self.opt.face_quality_thresh:
                    all_ok = False
                    break
                person_quality.append(quality_preds)

                # similarity to refs
                sims = [self.adaface_similarity.predict_similarity(entry['face'], single_face_img) for entry in best_faces]
                max_sim = max(sims) if sims else 0.0
                if not (0.6 <= max_sim <= 0.85):
                    all_ok = False
                    break
                person_sim.append(max_sim)
                best_ref_idx = int(np.argmax(sims))
                matched_refs.append(best_ref_idx)

            if not all_ok:
                continue

            avg_q = float(np.mean(person_quality)) if person_quality else 0.0
            avg_sim = float(np.mean(person_sim)) if person_sim else 0.0
            combined = combined_score_from_quality_and_sim(person_quality, person_sim) if (person_quality and person_sim) else (avg_q + avg_sim) / 2.0

            frame_scores.append({
                'frame_idx': frame_idx,
                'combined_score': combined,
                'matched_refs': matched_refs
            })

        # group and pick best per bin (bin size 150 frames as original)
        groups = defaultdict(list)
        for entry in frame_scores:
            bin_id = entry['frame_idx'] // 150
            groups[bin_id].append(entry)

        selected = []
        for bin_id, entries in groups.items():
            best = max(entries, key=lambda x: x['combined_score'])
            selected.append(best)
        selected.sort(key=lambda x: x['frame_idx'])

        # save refs and selected frames to out_dir, append jsonl entries
        os.makedirs(out_dir, exist_ok=True)
        ref_paths = []
        for i, entry in enumerate(best_faces):
            path = os.path.join(out_dir, f"ref_{i}.jpg")
            save_image(path, entry['body'])
            ref_paths.append(path)

        jsonl_path = self.opt.jsonl_path
        for entry in selected:
            idx = entry['frame_idx']
            try:
                frame = v.get_frame(idx)
            except Exception as e:
                logging.warning(f"Failed to fetch selected frame {idx}: {e}")
                continue
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            gt_fn = f"gt_{idx}.jpg"
            gt_path = os.path.join(out_dir, gt_fn)
            save_image(gt_path, bgr)
            input_images = [ref_paths[r] for r in entry['matched_refs']]
            record = {
                "task_type": "subject-driven",
                "instruction": "",
                "input_images": input_images,
                "output_image": gt_path
            }
            append_jsonl(jsonl_path, record)

        v.release()

    def __call__(self, item):
        """
        Keep compatibility with ray.map usage: Worker callable taking an item dict with 'file_path'.
        """
        vid_path = item.get('file_path') if isinstance(item, dict) else item
        try:
            self.process_video(vid_path)
        except Exception as e:
            logging.error(f"Error processing {vid_path}: {e}")
        return item


def main(opt):
    logging.basicConfig(
        filename=opt.log_path,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # local mode: read CSV and run sequentially
    if opt.is_local:
        samples = list(__import__("csv").DictReader(open(opt.input_csv_path, "r", encoding="utf-8-sig")))
        worker = Worker(opt)
        for item in samples:
            worker(item)
        return

    # distributed mode: use ray to map Worker over samples
    if ray is None:
        raise RuntimeError("Ray is not installed; either run with --is_local True or install ray.")

    # init ray (auto address expected in original script)
    ray.init(
        address="auto",
        runtime_env={
            "env_vars": {
                "PYTHONPATH": os.environ.get("PYTHONPATH", ""),
                "HF_ENDPOINT": os.environ.get("HF_ENDPOINT", ""),
                "DEEPFACE_HOME": os.environ.get("DEEPFACE_HOME", "")
            }
        }
    )

    samples = ray.data.read_csv(opt.input_csv_path)
    # map Worker over dataset: keep similar concurrency args as original (concurrency=num)
    # note: in ray 2.x, .map_batches or .map are options; we mimic original .map(Worker, num_gpus=1, concurrency=19)
    predictions = samples.map(Worker(opt), num_gpus=1, concurrency=19)
    # write out (original wrote to a fixed dir)
    out_csv_dir = opt.ray_output_dir or opt.ray_log_dir
    predictions.write_csv(out_csv_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='sam2 valid frame pipeline')
    parser.add_argument('--log_path', default='./log/log.log', help="日志地址")
    parser.add_argument('--jsonl_path', default='./out/out.jsonl')
    parser.add_argument('--output_dir_root', default='./out/')
    parser.add_argument('--input_csv_path', default='./input.csv')
    parser.add_argument('--is_local', default=True, type=lambda x: (str(x).lower() in ["1", "true", "yes"]))
    parser.add_argument("--ray_log_dir", type=str, default="./ray_log")
    parser.add_argument("--ray_output_dir", type=str, default=None)
    parser.add_argument("--det_checkpoint", default="")
    parser.add_argument("--det_config", default="")
    parser.add_argument("--face_quality_ckpt_path", default="")
    parser.add_argument("--face_quality_other_ckpt", default="")
    parser.add_argument("--face_quality_thresh", type=float, default=0.35)
    parser.add_argument("--face_match_thresh", type=float, default=0.5)
    opt = parser.parse_args()

    main(opt)
