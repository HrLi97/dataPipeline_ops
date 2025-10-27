# pipeline_scene.py
"""
简化版 pipeline:本地串行或 Ray 并行两种模式。
使用 ops 包中的 VideoProbeOp 和 SceneSegmenterOp 完成 probe + 切片工作。
"""

import os
import argparse
import logging
import csv
import ray
import pandas as pd

from common.video.video_probe_op import VideoProbeOp
from common.video.scene_segmenter_op import SceneSegmenterOp
from common.io.minio_upload_op import MinioUploadOp

def run_local(input_csv, output_csv, output_dir, min_duration_fps, segment_duration_sec, min_segment_duration_sec, log_path=None):
    if log_path:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        logging.basicConfig(filename=log_path, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    else:
        logging.basicConfig(level=logging.INFO)

    probe = VideoProbeOp()
    
    output_template = os.path.join(output_dir, "{base_name}", "{base_name}_{seg_idx:04d}.mp4")
    segmenter = SceneSegmenterOp(segment_duration_sec=segment_duration_sec,
                                 min_segment_duration_sec=min_segment_duration_sec,
                                 output_file_template=output_template)

    rows = list(csv.DictReader(open(input_csv, "r", encoding="utf-8-sig")))
    results = []
    for row in rows:
        file_path = row.get("file_path") or row.get("file_path[0]")
        item = {"file_path": file_path}
        logging.info(f"[LOCAL] 处理 {file_path}")

        item = probe.predict(item)
        total_dur = item.get("total_duration", 0.0)
        if total_dur <= 0 or total_dur < min_duration_fps:
            item["status"] = 0
            item["reason"] = "duration too short"
            results.append(item)
            continue

        item = segmenter.predict(item)
        item["status"] = 1
        results.append(item)

    # 写简要 CSV
    df = pd.DataFrame([{"file_path": r.get("file_path"), "status": r.get("status"), "segments_count": len(r.get("segments", []))} for r in results])
    df.to_csv(output_csv, index=False)
    logging.info(f"[LOCAL] 完成，结果写入 {output_csv}")


class Worker:
    """
    用于 ray.data.map_batches 的可调用类。
    在子进程首次调用时延迟初始化算子，避免序列化大对象。
    每个 batch(batch_size=1)返回列表形式的结果（可被 write_csv 写出）。
    """
    def __init__(self, output_dir, segment_duration_sec, min_segment_duration_sec, min_duration_fps, minio_cfg=None, log_path=None):
        self.output_dir = output_dir
        self.segment_duration_sec = segment_duration_sec
        self.min_segment_duration_sec = min_segment_duration_sec
        self.min_duration_fps = min_duration_fps
        self.minio_cfg = minio_cfg
        self.log_path = log_path
        # 不在 __init__ 中创建算子，避免序列化问题

    def _ensure_ops(self):
        # 在子进程中第一次调用时初始化
        if not hasattr(self, "probe"):
            self.probe = VideoProbeOp()

            output_template = os.path.join(self.output_dir, "{base_name}", "{base_name}_{seg_idx:04d}.mp4")
            self.segmenter = SceneSegmenterOp(segment_duration_sec=self.segment_duration_sec,
                                              min_segment_duration_sec=self.min_segment_duration_sec,
                                              output_file_template=output_template)
            if self.minio_cfg:
                try:
                    self.minio = MinioUploadOp(**self.minio_cfg)
                except Exception:
                    self.minio = None
            else:
                self.minio = None

    def __call__(self, batch):
        # batch 可能是 pandas.DataFrame（map_batches 默认）
        self._ensure_ops()
        rows = batch.to_dict(orient="records") if hasattr(batch, "to_dict") else list(batch)
        out = []
        for row in rows:
            file_path = row.get("file_path") or row.get("file_path[0]")
            item = {"file_path": file_path}
            logging.info(f"[RAY] 处理 {file_path}")

            item = self.probe.predict(item)
            total_dur = item.get("total_duration", 0.0)
            if total_dur <= 0 or total_dur < self.min_duration_fps:
                item["status"] = 0
                item["reason"] = "duration too short"
                out.append(item)
                continue

            item = self.segmenter.predict(item)
            if self.minio:
                item = self.minio.predict(item)
            item["status"] = 1
            out.append(item)
        return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_csv_path', default='/home/ubuntu/MyFiles/haoran/code/Data_process_talking/pipeline_for_i2i_person/data/r2v_batch1.csv')
    parser.add_argument('--output_dir', default='/home/ubuntu/MyFiles/haoran/code/data_source_all/i2i/batch-1/youtube/batch-1/cut-15min/')
    parser.add_argument('--log_path', default='/home/ubuntu/MyFiles/haoran/code/Data_process_talking/pipeline_for_i2i_person/data/log/log.log', help="日志地址")
    parser.add_argument('--max_duration_fps', default='208000', type=float)
    parser.add_argument('--min_duration_fps', default='30', type=float)
    parser.add_argument('--is_local', default=True, type=bool)
    parser.add_argument("--output_csv", type=str, default="./result.csv")
    parser.add_argument("--ray_log_dir", type=str, default="/code_home/Data_process_all/process_for_handthings/results/batch-2/log_dir/3_ray_log")
    parser.add_argument('--segment_duration_sec', default=900, type=int)
    parser.add_argument('--min_segment_duration_sec', default=30*5, type=int)
    args = parser.parse_args()

    if args.is_local:
        run_local(args.input_csv_path, args.output_csv, args.output_dir, args.min_duration_fps, args.segment_duration_sec, args.min_segment_duration_sec, args.log_path)
    else:
        # Ray 模式：保持跟原脚本一致的行为
        if not ray.is_initialized():
            ray.init()
        # 确保输出目录存在并设置权限（与原脚本一致）
        os.makedirs(os.path.dirname(args.ray_log_dir), exist_ok=True, mode=0o777)

        samples = ray.data.read_csv(args.input_csv_path)
        worker = Worker(output_dir=args.output_dir,
                        segment_duration_sec=args.segment_duration_sec,
                        min_segment_duration_sec=args.min_segment_duration_sec,
                        min_duration_fps=args.min_duration_fps,
                        minio_cfg=None,
                        log_path=args.log_path)

        predictions = samples.map_batches(
            worker,
            num_cpus=1,
            batch_size=1,
            concurrency=80,
        )
        predictions.write_csv(args.ray_log_dir)
        logging.info(f"[RAY] 完成，结果写入 {args.ray_log_dir}")