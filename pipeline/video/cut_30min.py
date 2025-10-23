"""
Pipeline that uses common.segmenter.cut_video_segments to perform the cut task.
This pipeline preserves the original CLI shape and supports:
 - local mode: iterate CSV lines, sequential processing
 - distributed (ray) mode: map Worker callable over CSV rows (map_batches)
It REUSES common modules:
  - common.segmenter.cut_video_segments
  - common.video_io.extract_video_info
  - common.io_ops.* (if needed)
  - common.ffmpeg_ops.cut_segment_ffmpeg (used by segmenter)
"""

import os
import sys
import csv
import logging
import argparse

# option: if running from repo root, ensure repo root in PYTHONPATH
# sys.path.append("/path/to/your/repo")  # adjust if needed

try:
    import ray
except Exception:
    ray = None

from common.segmenter import cut_video_segments

parser = argparse.ArgumentParser(description='scene slicing pipeline (fixed-length segmentation)')
parser.add_argument('--input_csv_path', default='/home/ubuntu/MyFiles/haoran/code/Data_process_talking/pipeline_for_i2i_person/data/r2v_batch1.csv')
parser.add_argument('--output_dir', default='/home/ubuntu/MyFiles/haoran/code/data_source_all/i2i/batch-1/youtube/batch-1/cut-15min/')
parser.add_argument('--log_path', default='/home/ubuntu/MyFiles/haoran/code/Data_process_talking/pipeline_for_i2i_person/data/log/log.log', help="日志地址")
parser.add_argument('--segment_duration', type=int, default=900)
parser.add_argument('--min_segment_duration', type=int, default=150)
parser.add_argument('--is_local', action='store_true', default=False)
parser.add_argument('--ffmpeg_retry', type=int, default=3)
parser.add_argument('--ray_log_dir', type=str, default="/code_home/Data_process_all/process_for_handthings/results/batch-2/log_dir/3_ray_log")
parser.add_argument('--concurrency', type=int, default=80)
opt = parser.parse_args()


class Worker:
    """
    Thin Worker callable that uses common.segmenter.cut_video_segments.
    Returns the original item (so Ray .map/.map_batches can write back results).
    """
    def __init__(self, output_dir, segment_duration, min_segment_duration, ffmpeg_retry):
        self.output_dir = output_dir
        self.segment_duration = segment_duration
        self.min_segment_duration = min_segment_duration
        self.ffmpeg_retry = ffmpeg_retry

    def __call__(self, item):
        # item may be a dict from CSV or [path] depending on how ray provides it
        if isinstance(item, dict):
            vid_path = item.get('file_path') or item.get('file')
        elif isinstance(item, (list, tuple)):
            vid_path = item[0]
        else:
            vid_path = item

        if not vid_path:
            return item

        try:
            processed, success_cnt, generated = cut_video_segments(
                video_path=vid_path,
                output_root=self.output_dir,
                segment_duration=self.segment_duration,
                min_segment_duration=self.min_segment_duration,
                ffmpeg_retry=self.ffmpeg_retry
            )
            # attach some metadata to item for downstream logging
            if isinstance(item, dict):
                item['processed'] = int(processed)
                item['segments_extracted'] = success_cnt
                item['generated_paths'] = ";".join(generated) if generated else ""
            else:
                # if item is simple value, return a dict
                item = {'file_path': vid_path, 'processed': int(processed), 'segments_extracted': success_cnt}
        except Exception as e:
            logging.exception(f"Error processing video {vid_path}: {e}")
            if isinstance(item, dict):
                item['processed'] = 0
                item['error'] = str(e)
            else:
                item = {'file_path': vid_path, 'processed': 0, 'error': str(e)}
        return item


def main():
    os.makedirs(os.path.dirname(opt.log_path), exist_ok=True, mode=0o777)
    logging.basicConfig(filename=opt.log_path, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    worker = Worker(
        output_dir=opt.output_dir,
        segment_duration=opt.segment_duration,
        min_segment_duration=opt.min_segment_duration,
        ffmpeg_retry=opt.ffmpeg_retry
    )

    if opt.is_local:
        samples = list(csv.DictReader(open(opt.input_csv_path, "r", encoding="utf-8-sig")))
        results = []
        for sample in samples:
            res = worker(sample)
            results.append(res)
        # optional: write a summary CSV next to log
        try:
            import pandas as pd
            pd.DataFrame(results).to_csv(os.path.join(os.path.dirname(opt.log_path), "cut_results_summary.csv"), index=False)
        except Exception:
            logging.exception("Failed to write results CSV")
        return

    # distributed mode (ray)
    if ray is None:
        raise RuntimeError("Ray not installed; run with --is_local or install ray")

    ray.init(runtime_env={"env_vars": {"PYTHONPATH": os.environ.get("PYTHONPATH", "")}})
    samples = ray.data.read_csv(opt.input_csv_path)

    # map_batches keeps same semantics as your original script
    predictions = samples.map_batches(
        Worker(opt.output_dir, opt.segment_duration, opt.min_segment_duration, opt.ffmpeg_retry),
        num_cpus=1,
        batch_size=1,
        concurrency=opt.concurrency
    )
    # write results to a directory, similar to original
    os.makedirs(os.path.dirname(opt.ray_log_dir), exist_ok=True, mode=0o777)
    predictions.write_csv(opt.ray_log_dir)


if __name__ == "__main__":
    main()
