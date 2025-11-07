import subprocess
import random
import time
import os
import logging  

from ..base_ops import BaseOps


class FFmpegCutOp(BaseOps):
    """
    使用 ffmpeg 进行视频切片的算子，包含重试逻辑。
    它支持通过 (start_time, duration) 或 (start_time, end_time) 来指定切片范围。
    """

    def __init__(self, max_retry: int = 3, ffmpeg_bin: str = "ffmpeg", **kwargs):
        self.max_retry = max_retry
        self.ffmpeg_bin = ffmpeg_bin

    def _run_ffmpeg(
        self, in_file: str, out_file: str, start: float, dur: float
    ) -> bool:
        """
        构建并执行 ffmpeg 命令。
        """
        cmd = [
            self.ffmpeg_bin,
            "-loglevel", "error",
            "-ss", str(start),
            "-i", in_file,
            "-t", str(dur),
            "-c:v", "libx264",  # 默认使用流拷贝，速度最快。如果需要重编码，可以改回 libx264
            "-c:a", "aac",
            "-map", "0",
            "-movflags", "+faststart",
            "-y", # 覆盖输出文件
            out_file,
        ]

        for attempt in range(self.max_retry):
            try:
                # 运行命令，并捕获标准输出和标准错误
                result = subprocess.run(
                    cmd, check=True, capture_output=True, text=True
                )
                logging.info(f"FFmpeg 切片成功: {out_file}")
                return True
            except subprocess.CalledProcessError as e:
                # 如果 ffmpeg 返回非零退出码，则会触发此异常
                wait = 2**attempt + random.random()
                logging.warning(
                    f"[WARN] ffmpeg 第 {attempt+1} 次尝试失败，将在 {wait:.1f}s 后重试。"
                    f"\nCMD: {' '.join(cmd)}"
                    f"\nStderr: {e.stderr}"
                )
                time.sleep(wait)
        
        logging.error(f"FFmpeg 切片在 {self.max_retry} 次尝试后最终失败: {out_file}")
        return False

    def predict(self, item: dict) -> dict:
        """
        执行切片逻辑。
        item 必须包含:
        - file_path: 输入视频路径
        - out_path: 输出视频路径
        - start_time: 切片开始时间（秒）
        并且必须包含以下两者之一:
        - duration: 持续时长（秒）
        - end_time: 切片结束时间（秒）
        """
        in_file = item.get("file_path")
        out_file = item.get("out_path")
        start = item.get("start_time", 0.0)
        
        # --- [核心修改逻辑] ---
        # 优先使用 duration，其次通过 end_time 计算 duration
        
        duration = item.get("duration")  # 获取 duration，如果不存在则为 None
        end_time = item.get("end_time")  # 获取 end_time，如果不存在则为 None
        
        final_duration = 0.0

        if duration is not None and duration > 0:
            # 1. 如果提供了有效的 duration，直接使用
            final_duration = duration
        elif end_time is not None and end_time > start:
            # 2. 如果没有提供 duration，但提供了有效的 end_time，则计算出 duration
            final_duration = end_time - start
        else:
            # 3. 如果无法确定 duration，则操作失败
            item["cut_error"] = "必须提供有效的 'duration' 或 'end_time' 以进行切片"
            item["cut_status"] = 0
            logging.error(f"{item['cut_error']} for file: {in_file}")
            return item

        if not in_file or not out_file:
            item["cut_error"] = "缺少输入文件或输出路径"
            item["cut_status"] = 0
            return item

        # 确保输出目录存在
        os.makedirs(os.path.dirname(out_file), exist_ok=True)

        # 调用 ffmpeg 执行切片
        success = self._run_ffmpeg(in_file, out_file, start, final_duration)
        
        item["cut_status"] = 1 if success else 0
        if not success:
            item["cut_error"] = "ffmpeg execution failed"
        
        return item
    
if __name__ == "__main__":
    csv_file_path = "/datas/workspace/wangshunyao/dataPipeline_ops/tmp/video_list.csv"
    
    with open(csv_file_path, 'r', encoding='utf-8') as f:
        video_path = f.readline().strip()

    cutter = FFmpegCutOp()
    test_case = {
        "file_path": video_path,
        "out_path": "/datas/workspace/wangshunyao/dataPipeline_ops/tmp/cut_videos_output/cut_from_csv.mp4",
        "start_time": 5.0,
        "duration": 3.5,
    }
    cutter.predict(test_case)