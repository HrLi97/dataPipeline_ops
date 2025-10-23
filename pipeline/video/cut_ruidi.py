# pipelines/cpu_worker/run.py  （或你原来的文件路径）
import os
import sys
import cv2
import subprocess
import pandas as pd
import ray
import time
import random
import argparse
import torch
import gc
import math
import logging

# constants kept same as original
MIN_SCENE_DURATION_SEC = 6
MAX_RETRY = 3
OUTPUT_DIR_DEFAULT = "/home/ubuntu/MyFiles/haoran/code/Data_process_talking/pipeline_for_i2i_person/data/out"
MIN_SEGMENT_DURATION_SEC = 30 * 5  # 30 * 5 seconds (as original)

# REUSE common.segmenter instead of repeating lower-level ops
from common.segmenter import cut_video_segments

class CpuWorker:
    # 固定分段时长（秒），保持你之前的设置
    SEGMENT_DURATION_SEC = 10 * 60  # 600s

    def __init__(self, output_root: str = OUTPUT_DIR_DEFAULT):
        self.output_root = output_root

    def cut_fixed_duration(self, video_file: str) -> int:
        """
        Use common.segmenter.cut_video_segments to cut into fixed-length segments.
        Returns the number of successfully created segments.
        """
        try:
            # 调用 segmenter，显式传入 segment_duration 与 min_segment_duration 保持行为不变
            processed, success_count, generated = cut_video_segments(
                video_path=video_file,
                output_root=self.output_root,
                segment_duration=self.SEGMENT_DURATION_SEC,
                min_segment_duration=MIN_SEGMENT_DURATION_SEC,
                ffmpeg_retry=MAX_RETRY
            )
            return success_count
        except Exception as e:
            logging.exception(f"cut_video_segments failed for {video_file}: {e}")
            return 0

    def __call__(self, item):
        data = {}
        video_path = item['file_path'] if isinstance(item, dict) else item
        print(video_path, "video_pathvideo_pathvideo_pathvideo_path")
        cnt = self.cut_fixed_duration(video_path)
        # 保留原来行为，标记 status
        try:
            item['status'] = 1
            item['segments_extracted'] = cnt
        except Exception:
            # 如果 item 不是 dict，返回 dict
            item = {'file_path': video_path, 'status': 1, 'segments_extracted': cnt}
        # cleanup some memory
        gc.collect()
        return item


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--is_local", action="store_true", default=False)
    parser.add_argument("--csv_path", type=str, default="/home/ubuntu/MyFiles/haoran/code/Data_process_talking/pipeline_for_i2i_person/data/r2v_batch1.csv")
    parser.add_argument("--output_csv", type=str, default="./result.csv")
    parser.add_argument("--output_log", type=str, default="/home/ubuntu/MyFiles/haoran/code/data_source_all/i2i/batch-1/youtube/batch-1/cut-10min/")
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR_DEFAULT)
    parser.add_argument("--concurrency", type=int, default=80)
    args = parser.parse_args()

    # ensure output root is picked up by CpuWorker
    worker = CpuWorker(output_root=args.output_dir)

    if args.is_local:
        samples = list(pd.read_csv(args.csv_path).T.to_dict().values())
        results = []
        for item in samples:
            print(item, "itemitemitemitem")
            results.append(worker(item))
        print(results, "resultsresultsresultsresultsresultsresults")
        # optionally save results to CSV
        try:
            df = pd.DataFrame(results)
            df.to_csv(args.output_csv, index=False)
        except Exception as e:
            logging.warning(f"Failed to write output CSV: {e}")
        return

    # distributed mode with Ray
    ray.init(
        runtime_env={
            "env_vars": {
                "PYTHONPATH": ":".join([
                    "/home/ubuntu/MyFiles/ruidi/Multi-Person-R2V/transnetv2pt"
                ])
            }
        }
    )
    samples = ray.data.read_csv(args.csv_path)
    # map Worker over dataset
    # IMPORTANT: be careful with concurrency -> too many parallel ffmpeg will exhaust system
    predictions = samples.map(
        CpuWorker(args.output_dir),
        num_cpus=1,
        concurrency=args.concurrency,
    )
    predictions.write_csv(args.output_log)


if __name__ == "__main__":
    main()
