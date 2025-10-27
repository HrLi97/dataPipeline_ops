"""
使用 ffprobe 从视频文件中获取精确的时长（秒）和 fps。
"""
import subprocess
from ..base_ops import BaseOps

class VideoProbeOp(BaseOps):
    """
    通过 ffprobe 获取视频的 duration 和 fps,并写回 item。
    item 入参：{"file_path": "/path/to/video.mp4"}。
    item 出参会包含 "total_duration" (float 秒) 和 "fps" (float)，
    以及在出错时附加 "probe_error" 字段。
    """

    def __init__(self, ffprobe_bin: str = "ffprobe", **kwargs):
        self.ffprobe_bin = ffprobe_bin

    def _get_duration(self, video_path: str):
        cmd = [
            self.ffprobe_bin, "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            video_path
        ]
        try:
            r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            return float(r.stdout)
        except Exception as e:
            return None

    def _get_fps(self, video_path: str):
        # 读取 video stream 的 avg_frame_rate 或 r_frame_rate
        cmd = [
            self.ffprobe_bin, "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=avg_frame_rate,r_frame_rate",
            "-of", "default=noprint_wrappers=1:nokey=1",
            video_path
        ]
        try:
            r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            text = r.stdout.decode().strip().splitlines()
            # 优先取第一行（avg_frame_rate），格式可能为 '30000/1001'
            for line in text:
                if '/' in line:
                    num, den = line.split('/')
                    try:
                        return float(num) / float(den)
                    except:
                        continue
                else:
                    try:
                        return float(line)
                    except:
                        continue
            return 0.0
        except Exception:
            return 0.0

    def predict(self, item: dict) -> dict:
        path = item.get("file_path")
        if not path:
            item["probe_error"] = "no file_path provided"
            item["total_duration"] = 0.0
            item["fps"] = 0.0
            return item

        duration = self._get_duration(path)
        fps = self._get_fps(path)
        if duration is None:
            item["probe_error"] = f"ffprobe failed for {path}"
            item["total_duration"] = 0.0
            item["fps"] = fps or 0.0
        else:
            item["total_duration"] = float(duration)
            item["fps"] = float(fps or 0.0)
        return item
