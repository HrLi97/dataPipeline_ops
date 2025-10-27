import os
from ..base_ops import BaseOps

DEFAULT_OUTPUT_DIR = "/tmp/pipeline_out"

class GenerateSavePathOp(BaseOps):
    """
    输出路径生成算子：根据视频名、段索引、开始时间和时长生成 out_path 并创建目录。
    """

    def __init__(self, output_dir: str = DEFAULT_OUTPUT_DIR, **kwargs):
        self.output_dir = output_dir

    def predict(self, item: dict) -> dict:
        video_file = item.get("file_path")
        seg_idx = item.get("seg_idx", 0)
        start_time = item.get("start_time", 0)
        duration = item.get("duration", 0)

        if not video_file:
            item["save_path_error"] = "没有提供 file_path"
            return item

        base = os.path.basename(video_file)
        name, _ = os.path.splitext(base)
        save_dir = os.path.join(self.output_dir, name)
        os.makedirs(save_dir, exist_ok=True)

        out_path = os.path.join(save_dir, f"{name}_seg{seg_idx}_{int(start_time)}_{int(duration)}.mp4")
        item["out_path"] = out_path
        item["save_dir"] = save_dir
        return item
