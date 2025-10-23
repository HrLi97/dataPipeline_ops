# common/segmenter.py
"""
Segmenter utilities that reuse common.io_ops and common.ffmpeg_ops.
- Uses: common.video_io.extract_video_info
        common.io_ops.generate_output_path
        common.ffmpeg_ops.cut_segment_ffmpeg
"""

from datetime import timedelta
from typing import List, Tuple
import os
import logging

# eager imports at top (per你的要求)
from common.video_io import extract_video_info
from common.io_ops import generate_output_path
from common.ffmpeg_ops import cut_segment_ffmpeg

# default constants
DEFAULT_SEGMENT_DURATION_SEC = 900  # 15 minutes
DEFAULT_MIN_SEGMENT_DURATION_SEC = 30 * 5  # 150s


def split_scene_recursively(scene: Tuple[int, int], fps: float, max_duration: int, min_duration: int) -> List[Tuple[int, int]]:
    start_frame, end_frame = scene
    scene_duration = end_frame - start_frame

    if min_duration <= scene_duration <= max_duration:
        return [scene]

    if scene_duration > max_duration:
        mid_frame = (start_frame + end_frame) // 2
        left = (start_frame, mid_frame)
        right = (mid_frame + 1, end_frame)
        return (
            split_scene_recursively(left, fps, max_duration, min_duration) +
            split_scene_recursively(right, fps, max_duration, min_duration)
        )
    return []

def cut_video_segments(
    video_path: str,
    output_root: str,
    segment_duration: int = DEFAULT_SEGMENT_DURATION_SEC,
    min_segment_duration: int = DEFAULT_MIN_SEGMENT_DURATION_SEC,
    ffmpeg_retry: int = None
) -> Tuple[bool, int, List[str]]:
    """
    Cut video into fixed-length segments and write them to disk.

    Returns: (processed: bool, success_count: int, generated_paths: List[str])
    """

    if ffmpeg_retry is None:
        ffmpeg_retry = 3

    fps, total_duration = extract_video_info(video_path)
    if not total_duration or total_duration <= 0:
        logging.warning(f"Unable to determine duration for {video_path}")
        return False, 0, []

    if total_duration < min_segment_duration:
        logging.info(f"Video too short ({total_duration}s) for segmentation: {video_path}")
        return False, 0, []

    segments = []
    current_start = 0.0
    while current_start < total_duration:
        end_time = min(current_start + segment_duration, total_duration)
        if (end_time - current_start) >= min_segment_duration:
            segments.append((current_start, end_time))
        current_start = end_time

    # ensure output dir exists
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    os.makedirs(os.path.join(output_root, video_name), exist_ok=True)

    success_count = 0
    generated = []

    for seg_idx, (start_sec, end_sec) in enumerate(segments, start=1):
        # now reuse common.io_ops.generate_output_path
        out_path = generate_output_path(output_root, video_path, seg_idx, start_sec, (end_sec - start_sec))

        ok, logs = cut_segment_ffmpeg(
            video_file=video_path,
            out_path=out_path,
            start_sec=start_sec,
            seg_dur=(end_sec - start_sec),
            max_retry=ffmpeg_retry
        )
        if ok:
            success_count += 1
            generated.append(out_path)
            logging.info(f"Segment generated: {out_path}")
        else:
            for ln in logs:
                logging.warning(f"Segment failure for {video_path}: {ln}")

    processed = success_count > 0
    return processed, success_count, generated
