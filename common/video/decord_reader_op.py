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

class DecordReaderOp(BaseOps):
    def __init__(self, use_gpu=False, **kwargs):
        self.use_gpu = use_gpu

    def _open_vr(self, video_path):
        # 使用 ffmpeg.probe 获取宽高并指定 decord 的 width/height 避免尺寸变化
        meta = ffmpeg.probe(video_path)
        vs = next(s for s in meta['streams'] if s['codec_type']=='video')
        width, height = int(vs['width']), int(vs['height'])
        ctx = cpu(0)
        vr = VideoReader(video_path, width=width, height=height, ctx=ctx)
        return vr

    def predict(self, data: VideoData) -> VideoData:
        # 精简后的数据类仅包含 tensor/format/scale；视频路径/reader 不再挂在对象上
        # 此算子在精简模式下不执行文件读取，建议在外部把帧转换为 tensor 后传入本算子链路
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

if __name__ == "__main__":
    csv_file_path = "/datas/workspace/wangshunyao/dataPipeline_ops/tmp/video_list.csv"
    with open(csv_file_path, 'r', encoding='utf-8') as f:
        video_paths = [line.strip() for line in f if line.strip()]
    reader = DecordReaderOp()
    for video_path in video_paths:
        item = {"file_path": video_path}
        item = reader.predict(item)
        
        vr = item["vr"]
        fps = vr.get_avg_fps()
        idxs, frames = reader.sample_frames_by_seconds(vr, fps, interval_sec=5)
        
        idxs_ref, frames_ref = reader.sample_ref_frames_last_n_seconds(vr, fps, last_n_seconds=1)
