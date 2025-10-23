# common/ffmpeg_ops.py
import subprocess
import time
import random
from typing import Tuple, List

MAX_RETRY = 3

def cut_segment_ffmpeg(
    video_file: str,
    out_path: str,
    start_sec: float,
    seg_dur: float,
    codec: str = "libx264",
    crf: int = 18,
    preset: str = "fast",
    pix_fmt: str = "yuv420p",
    audio_codec: str = "aac",
    movflags: str = "+faststart",
    max_retry: int = MAX_RETRY,
) -> Tuple[bool, List[str]]:
    """
    Use ffmpeg to cut a single segment from `video_file` to `out_path`.
    Returns (success: bool, logs: [stderr_lines]).
    Retries up to max_retry with exponential backoff + jitter.
    """
    cmd = [
        "ffmpeg", "-loglevel", "error", "-y", "-i", video_file,
        "-ss", str(start_sec), "-t", str(seg_dur),
        "-c:v", codec, "-crf", str(crf), "-preset", preset,
        "-pix_fmt", pix_fmt, "-c:a", audio_codec,
        "-movflags", movflags,
        out_path
    ]

    logs = []
    for attempt in range(max_retry):
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return True, logs
        except Exception as e:
            wait = 2 ** attempt + random.random()
            logs.append(f"Attempt {attempt} failed: {e}")
            time.sleep(wait)
    logs.append(f"All {max_retry} attempts failed for segment start={start_sec}, dur={seg_dur}")
    return False, logs
