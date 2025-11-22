# ops/decord_reader_op.py
"""
使用 decord 读取视频并提供抽帧接口（比如按每 5s 取帧、取最后 1s 的参考帧等）。
"""
from decord import VideoReader, cpu
import ffmpeg
import numpy as np
import os
import sys

from ..base_ops import BaseOps
from ..data_models import VideoData
from ..data_models import ImageData,FFmpegVideoInfo


class DecordReaderOp(BaseOps):
    def __init__(self, use_gpu=False, **kwargs):
        self.use_gpu = use_gpu

    def predict(self, data: VideoData) -> VideoData:
        return data

    # 辅助方法：按秒抽帧（每 interval_sec 抽一帧），返回 ndarray list（RGB）
    def sample_frames_by_seconds(self, vr, fps, interval_sec=5):
        total_frame = len(vr)
        step = max(1, int(round(interval_sec * fps)))
        idxs = list(range(0, total_frame, step))
        frames = vr.get_batch(idxs).asnumpy()
        return idxs, frames

    # 获取最后 1s 的若干参考帧（按 1/秒 抽）
    def sample_ref_frames_last_n_seconds(self, vr, fps, last_n_seconds=1):
        total_frame = len(vr)
        n_frames = max(1, int(round(last_n_seconds * fps)))
        start = max(0, total_frame - n_frames)
        idxs = list(range(start, total_frame))
        frames = vr.get_batch(idxs).asnumpy()
        return idxs, frames

    csv_file_path = "/mnt/cfs/shanhai/lihaoran/project/code/dataPipeline_ops/tmp/video_list.csv"
    with open(csv_file_path, 'r', encoding='utf-8') as f:
        video_paths = [line.strip() for line in f if line.strip()]
    reader = DecordReaderOp()
    for video_path in video_paths:
        print(video_path)
        vi = FFmpegVideoInfo(video_path)
        print( "num_frame:", vi.num_frame)
        vd = vi.load_data(offset=0, max_num_frame=min(5, vi.num_frame), sample_rate=max(1, int(vi.fps) or 1))
        if vd.data is None:
            print("no frames read")
            continue
        print("format:", vd.format, "shape:", vd.shape)
        # safe access first frame in NHWC for inspection
        frame0 = vd.data[0] if vd.format.upper() == "NHWC" else vd.data[0].permute(1, 2, 0)
        print("frame0 shape:", tuple(frame0.shape))
        
        # item = reader.predict(item)
        
        # vr = item["vr"]
        # fps = vr.get_avg_fps()
        # idxs, frames = reader.sample_frames_by_seconds(vr, fps, interval_sec=5)
        
        # idxs_ref, frames_ref = reader.sample_ref_frames_last_n_seconds(vr, fps, last_n_seconds=1)
