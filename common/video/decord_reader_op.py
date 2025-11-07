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

    def predict(self, item: dict) -> dict:
        path = item.get("file_path")
        if not path:
            item["reader_error"] = "no file_path"
            return item
        try:
            vr = self._open_vr(path)
            total_frame = len(vr)
            item["vr"] = vr  # 注意：如果在多进程中传递 item，这个对象不可序列化；仅供本地流程使用
            item["total_frame"] = total_frame
            item["height"] = vr[0].asnumpy().shape[0]
            item["width"] = vr[0].asnumpy().shape[1]
        except Exception as e:
            item["reader_error"] = str(e)
        return item

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

    if not os.path.exists(csv_file_path):
        print(f"错误:CSV 文件不存在于 -> {csv_file_path}")
        sys.exit(1)

    with open(csv_file_path, 'r', encoding='utf-8') as f:
        video_paths = [line.strip() for line in f if line.strip()]

    if not video_paths:
        print("错误:CSV 文件中没有找到任何视频路径。")
        sys.exit(1)
    
    print(f"成功从 CSV 文件中读取 {len(video_paths)} 个视频路径。")

    reader = DecordReaderOp()
    for video_path in video_paths:
        print(f"\n======== 正在处理: {video_path} ========")

        if not os.path.exists(video_path):
            print(f"警告：跳过，文件不存在 -> {video_path}")
            continue

        item = {"file_path": video_path}
        item = reader.predict(item)
        
        if "reader_error" in item:
            print("错误:", item["reader_error"])
        else:
            vr = item["vr"]
            fps = vr.get_avg_fps()
            print("总帧数:", item["total_frame"], "| FPS:", fps)
            
            idxs, frames = reader.sample_frames_by_seconds(vr, fps, interval_sec=5)
            print("按5秒采样索引:", idxs)
            
            idxs_ref, frames_ref = reader.sample_ref_frames_last_n_seconds(vr, fps, last_n_seconds=1)
            print("最后1秒参考帧索引:", idxs_ref)