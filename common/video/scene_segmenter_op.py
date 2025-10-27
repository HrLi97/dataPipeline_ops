"""
计算分段并使用 scenedetect.video_splitter.split_video_ffmpeg 来切片。
"""

import os
import logging
from datetime import timedelta

from scenedetect.frame_timecode import FrameTimecode
from scenedetect.video_splitter import split_video_ffmpeg

from ..base_ops import BaseOps

# 默认的 FFmpeg 参数，用于拷贝视频流、音频流和字幕流，避免重新编码
DEFAULT_FFMPEG_ARGS = "-map 0:v:0 -map 0:a? -map 0:s? -c copy"

def split_scene_recursively(scene, fps, max_duration, min_duration):
    """
    递归地将场景拆分为符合时长要求的片段(保留原始实现，供可能的场景检测分段使用)。
    scene: (start_frame, end_frame)
    返回值：[(start_frame, end_frame), ...]
    """
    start_frame, end_frame = scene
    scene_duration = end_frame - start_frame

    if min_duration <= scene_duration <= max_duration:
        return [scene]

    if scene_duration > max_duration:
        mid_frame = (start_frame + end_frame) // 2
        left = (start_frame, mid_frame)
        right = (mid_frame + 1, end_frame)
        return split_scene_recursively(
            left, fps, max_duration, min_duration
        ) + split_scene_recursively(right, fps, max_duration, min_duration)
    return []


class SceneSegmenterOp(BaseOps):
    """
    负责按固定时长对视频进行分段，并调用 ffmpeg 进行切片。

    配置参数：
        segment_duration_sec: 每段目标时长（秒）。
        min_segment_duration_sec: 最小接受的段长（秒），最后一段如果小于此时长将被舍弃。
        output_file_template: 输出文件的路径模板。
            可用的占位符:
            - {base_name}: 不含扩展名的原始视频文件名
            - {start_time}: 形如 "HH-MM-SS" 的开始时间
            - {end_time}: 形如 "HH-MM-SS" 的结束时间
            - {seg_idx}: 切片的序号 (从 1 开始)
            - {start_sec}: 开始时间的总秒数
            - {end_sec}: 结束时间的总秒数
        arg_override: 传递给 split_video_ffmpeg 的 ffmpeg 参数字符串。

    输出：在 item 中填充 "segments" 列表，列表元素包含 out_path、start_time、end_time、status 等信息。
    """

    def __init__(
        self,
        segment_duration_sec: int = 900,
        min_segment_duration_sec: int = 300,
        output_file_template: str = "/tmp/pipeline_out/{base_name}/{base_name}_{seg_idx:04d}.mp4",
        arg_override: str = DEFAULT_FFMPEG_ARGS,
    ):
        self.segment_duration_sec = segment_duration_sec
        self.min_segment_duration_sec = min_segment_duration_sec
        self.output_file_template = output_file_template
        self.arg_override = arg_override

    def _time_to_label(self, t: float) -> str:
        # 将秒转换为 00-00-00 格式的字符串
        return str(timedelta(seconds=int(t))).replace(":", "-")

    def predict(self, item: dict) -> dict:
        """
        处理单个视频项。
        item 必须包含 file_path, total_duration, fps 字段（建议先通过 VideoProbeOp 填充）。
        """
        video_path = item.get("file_path")
        total_duration = item.get("total_duration", 0.0)
        fps = item.get("fps", None)

        if not video_path or not os.path.exists(video_path):
            item["segments"] = []
            item["segment_error"] = f"file_path not found or is empty: {video_path}"
            logging.warning(item["segment_error"])
            return item

        if total_duration <= 0 or fps is None:
            item["segments"] = []
            item["segment_error"] = f"invalid metadata: total_duration={total_duration}, fps={fps}"
            logging.warning(item["segment_error"])
            return item

        # 根据固定时长计算分段的时间点
        segments_times = []
        cur = 0.0
        while cur < total_duration:
            end = min(cur + self.segment_duration_sec, total_duration)
            # 只有当片段时长大于等于阈值时才保留
            if (end - cur) >= self.min_segment_duration_sec:
                segments_times.append((cur, end))
            else:
                logging.info(f"Skipping final segment, duration {end - cur:.2f}s is less than min_segment_duration_sec {self.min_segment_duration_sec}s.")
            cur = end

        segments = []
        base_name = os.path.splitext(os.path.basename(video_path))[0]

        for idx, (start_sec, end_sec) in enumerate(segments_times, 1):
            start_str = self._time_to_label(start_sec)
            end_str = self._time_to_label(end_sec)
            
            # 使用模板生成输出路径
            out_path = self.output_file_template.format(
                base_name=base_name,
                start_time=start_str,
                end_time=end_str,
                seg_idx=idx,
                start_sec=start_sec,
                end_sec=end_sec
            )
            
            # 确保输出目录存在
            output_dir = os.path.dirname(out_path)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
                logging.info(f"Created output directory: {output_dir}")

            # 调用 scenedetect.split_video_ffmpeg 进行切片
            status = 0
            try:
                split_video_ffmpeg(
                    video_path,
                    scene_list=[
                        (
                            FrameTimecode(timecode=start_sec, fps=fps),
                            FrameTimecode(timecode=end_sec, fps=fps),
                        )
                    ],
                    output_file_template=out_path,  # 这里传递的是最终的、完整的文件路径
                    arg_override=self.arg_override,
                    show_progress=False,
                    suppress_output=True # 抑制ffmpeg的控制台输出
                )
                if os.path.exists(out_path):
                    logging.info(f"Successfully split segment: {out_path}")
                    status = 1
                else:
                    logging.error(f"Failed to split segment (file not created): {out_path}")

            except Exception as e:
                logging.error(f"Error splitting segment {out_path}: {e}")
                status = 0

            segments.append(
                {
                    "seg_idx": idx,
                    "start_time": start_sec,
                    "end_time": end_sec,
                    "duration": end_sec - start_sec,
                    "out_path": out_path,
                    "status": status,
                }
            )

        item["segments"] = segments
        return item