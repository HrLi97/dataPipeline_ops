import math
import os
import argparse
import pandas as pd
import ray
from typing import Any, Dict

from common.video.video_info_op import VideoInfoOp
from common.io.generate_save_path_op import GenerateSavePathOp
from common.video.ffmpeg_cut_op import FFmpegCutOp

"""
- 本地模式：使用 pandas 逐条调用 CpuWorker()
- 非本地模式：使用 Ray, 直接传类 CpuWorker 给 samples.map
"""

DEFAULT_OUTPUT_DIR = "/datas/workspace/wangshunyao/Datasets/cartoon test/dataset/batch_3_cut/pipeline_out_1"


class CpuWorker:
    """
    使用原子算子完成切片任务: VideoInfoOp、GenerateSavePathOp、FFmpegCutOp
    该类可序列化为 Ray 的任务（在子进程中构造算子实例）。
    """

    SEGMENT_DURATION_SEC = 10 * 60  # 默认分段时长 10 分钟

    def __init__(self):
        # 在实例化时创建算子（在本地模式或子进程中都会执行）
        self.video_info_op = VideoInfoOp()
        self.path_op = GenerateSavePathOp(output_dir=DEFAULT_OUTPUT_DIR)
        self.ffmpeg_op = FFmpegCutOp(max_retry=3)

    def _cut_video_by_fixed_duration(
        self, video_file: str, segment_duration: int, min_segment_duration: int
    ):
        """
        按固定时长切分视频的实现（核心逻辑）。
        返回 segments 列表，每个元素为 dict 包含 out_path、start_time、duration、cut_status。
        """
        item = {"file_path": video_file}
        # 1) 提取视频信息
        item = self.video_info_op.predict(item)
        total_dur = float(item.get("total_duration", 0.0))
        fps = item.get("fps", None)

        if total_dur <= 0 or total_dur < min_segment_duration:
            return []

        num_segs = math.ceil(total_dur / segment_duration)
        segments = []
        for idx in range(num_segs):
            start_sec = idx * segment_duration
            seg_dur = min(segment_duration, total_dur - start_sec)
            if seg_dur < min_segment_duration:
                break

            seg_item = {
                "file_path": video_file,
                "seg_idx": idx,
                "start_time": start_sec,
                "duration": seg_dur,
            }
            # 2) 生成输出路径
            seg_item = self.path_op.predict(seg_item)
            # 3) 调用 ffmpeg 切片
            seg_item = self.ffmpeg_op.predict(seg_item)

            segments.append(
                {
                    "seg_idx": idx,
                    "start_time": start_sec,
                    "duration": seg_dur,
                    "out_path": seg_item.get("out_path"),
                    "cut_status": int(seg_item.get("cut_status", 0)),
                }
            )
        return segments

    def __call__(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """
        接收单个 item(dict)，并返回处理后的 item（包含 status / segments / reason）。
        兼容本地 CSV 行结构或 ray.data.map 的输入结构（保持与原脚本一致）。
        """
        try:
            # 兼容一些 CSV 行格式：file_path 可能是字符串或列表等
            video_path = item.get("file_path") if isinstance(item, dict) else item
            if isinstance(video_path, (list, tuple)):
                # 若是 list，取第一个元素
                video_path = video_path[0] if len(video_path) > 0 else None

            if not video_path:
                return {"file_path": video_path, "status": 0, "reason": "no file_path"}

            print(video_path, "video_pathvideo_pathvideo_pathvideo_path")
            segments = self._cut_video_by_fixed_duration(
                video_file=video_path,
                segment_duration=self.SEGMENT_DURATION_SEC,
                min_segment_duration=30 * 5,
            )

            result = {
                "file_path": video_path,
                "status": 1 if len(segments) > 0 else 0,
                "segments": segments,
            }
            return result
        except Exception as e:
            return {
                "file_path": item.get("file_path") if isinstance(item, dict) else None,
                "status": 0,
                "reason": str(e),
            }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--is_local", action="store_true", default=False)
    parser.add_argument(
        "--csv_path",
        type=str,
        default="/home/ubuntu/MyFiles/haoran/code/Data_process_talking/pipeline_for_i2i_person/data/r2v_batch1.csv",
    )
    parser.add_argument("--output_csv", type=str, default="./result.csv")
    parser.add_argument(
        "--output_log",
        type=str,
        default="/home/ubuntu/MyFiles/haoran/code/data_source_all/i2i/batch-1/youtube/batch-1/cut-10min/",
    )
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--segment_duration_sec", type=int, default=10 * 60)
    parser.add_argument("--min_segment_duration_sec", type=int, default=30 * 5)
    args = parser.parse_args()

    # 本地模式：用 pandas 读取 CSV，逐条调用 CpuWorker()
    if args.is_local:
        samples = list(pd.read_csv(args.csv_path).T.to_dict().values())
        worker = CpuWorker()
        results = []
        for item in samples:
            print("[INFO] 处理:", item.get("file_path"))
            res = worker(item)
            results.append(res)

        # 把简要结果写入 CSV（保留 file_path / status / segments_count）
        df = pd.DataFrame(
            [
                {
                    "file_path": r.get("file_path"),
                    "status": r.get("status"),
                    "segments": len(r.get("segments", [])) if r.get("segments") else 0,
                }
                for r in results
            ]
        )
        df.to_csv(args.output_csv, index=False)
        print(f"[INFO] 本地模式处理完成，结果已写入 {args.output_csv}")

    else:
        # 非本地模式：恢复为原始 Ray 流程，直接把类传入 samples.map
        ray.init(
            runtime_env={
                "env_vars": {
                    # 保留你原先的 PYTHONPATH 设置（如需要）
                    "PYTHONPATH": ":".join(
                        ["/home/ubuntu/MyFiles/ruidi/Multi-Person-R2V/transnetv2pt"]
                    )
                }
            }
        )

        samples = ray.data.read_csv(args.csv_path)

        # 直接把类传入 map，使 Ray 在每个进程/任务中执行 CpuWorker()
        predictions = samples.map(
            CpuWorker,
            num_cpus=1,
            concurrency=80,
        )

        # 确保输出目录存在，再把结果写成多个 CSV 文件（每个分区一个文件）
        os.makedirs(args.output_log, exist_ok=True)
        predictions.write_csv(args.output_log)
        print(f"[INFO] Ray 模式处理完成，结果已写入 {args.output_log}")


if __name__ == "__main__":
    main()
