import subprocess
from ..base_ops import BaseOps

class VideoProbeOp(BaseOps):
    """
    使用 ffprobe 获取视频的时长（秒）和 fps。
    item 输入需包含 "file_path"。
    结果写回 item["total_duration"], item["fps"]。出错写 item["probe_error"]。
    """

    def __init__(self, ffprobe_bin: str = "ffprobe", **kwargs):
        self.ffprobe_bin = ffprobe_bin

    def _get_duration(self, path):
        cmd = [self.ffprobe_bin, "-v", "error",
               "-show_entries", "format=duration",
               "-of", "default=noprint_wrappers=1:nokey=1", path]
        try:
            r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            return float(r.stdout)
        except Exception as e:
            return None

    def _get_fps(self, path):
        cmd = [self.ffprobe_bin, "-v", "error", "-select_streams", "v:0",
               "-show_entries", "stream=avg_frame_rate", "-of", "default=noprint_wrappers=1:nokey=1", path]
        try:
            r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            text = r.stdout.decode().strip()
            if '/' in text:
                a,b = text.split('/')
                return float(a)/float(b)
            else:
                return float(text)
        except Exception:
            return 0.0

    def predict(self, item: dict) -> dict:
        path = item.get("file_path")
        if not path:
            item["probe_error"] = "no file_path"
            item["total_duration"] = 0.0
            item["fps"] = 0.0
            return item
        duration = self._get_duration(path)
        fps = self._get_fps(path)
        if duration is None:
            item["probe_error"] = f"ffprobe failed for {path}"
            item["total_duration"] = 0.0
        else:
            item["total_duration"] = float(duration)
        item["fps"] = float(fps or 0.0)
        return item

if __name__ == "__main__":
    csv_file_path = "/datas/workspace/wangshunyao/dataPipeline_ops/tmp/video_list.csv"
    
    with open(csv_file_path, 'r', encoding='utf-8') as f:
        video_path = f.readline().strip()

    probe_op = VideoProbeOp()
    test_item = {"file_path": video_path}
    result_item = probe_op.predict(test_item)
    
    print(result_item)