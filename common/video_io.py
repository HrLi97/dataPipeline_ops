# common/video_io.py
from typing import Sequence, Dict, Any, Tuple
import numpy as np
from decord import VideoReader, cpu
import ffmpeg
import cv2
import os

class VideoIO:
    """Wrapper for decord + ffmpeg metadata probe."""
    def __init__(self, path: str, use_gpu: bool=False):
        self.path = path
        self.use_gpu = use_gpu
        # use ffmpeg.probe for accurate video stream metadata
        meta = ffmpeg.probe(path)
        vstream = next(s for s in meta['streams'] if s['codec_type']=='video')
        self.width = int(vstream['width']); self.height = int(vstream['height'])
        # Use cpu ctx by default to avoid GPU locking on import
        self.vr = VideoReader(path, width=self.width, height=self.height, ctx=cpu(0))
        self.total_frames = len(self.vr)
        # decord returns fps possibly as numpy type, cast to float
        try:
            self.fps = float(self.vr.get_avg_fps())
        except Exception:
            # fallback: try cv2 if decord can't get fps
            fps, _ = extract_video_info(path)
            self.fps = float(fps) if fps else 0.0

    def metadata(self) -> Dict[str, Any]:
        return {'width': self.width, 'height': self.height, 'total_frames': self.total_frames, 'fps': self.fps}

    def get_frame(self, idx: int) -> np.ndarray:
        f = self.vr[idx]
        if hasattr(f, 'asnumpy'):
            return f.asnumpy()
        return f.numpy()

    def get_batch(self, indices: Sequence[int]) -> np.ndarray:
        return self.vr.get_batch(indices).asnumpy()

    def release(self):
        try:
            self.vr.close()
        except Exception:
            pass


def extract_video_info(video_file: str) -> Tuple[float, float]:
    """
    Light-weight helper to get fps and total duration in seconds using cv2.
    This mirrors the previous `video_utils.extract_video_info`.
    Returns (fps, total_duration_seconds).
    """
    if not os.path.exists(video_file):
        return 0.0, 0.0
    cap = cv2.VideoCapture(video_file)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    total_duration = frame_count / fps if fps > 0 else 0.0
    return fps, total_duration
