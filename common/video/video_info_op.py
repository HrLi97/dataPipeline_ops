# ops/video_info_op.py
import os
import cv2
from ..base_ops import BaseOps

class VideoInfoOp(BaseOps):
    """
    视频信息提取算子：提取 fps、帧数和时长等信息。
    """

    def __init__(self, **kwargs):
        # 未来可加入更多配置项，如选择后端等
        self._cfg = kwargs

    def predict(self, item: dict) -> dict:
        video_file = item.get("file_path")
        if not video_file or not os.path.exists(video_file):
            item["video_info_error"] = f"文件不存在: {video_file}"
            item["fps"] = 0.0
            item["frame_count"] = 0
            item["total_duration"] = 0.0
            return item

        cap = cv2.VideoCapture(video_file)
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
        cap.release()
        total_duration = frame_count / fps if fps > 0 else 0.0

        item["fps"] = float(fps)
        item["frame_count"] = int(frame_count)
        item["total_duration"] = float(total_duration)
        return item

if __name__ == "__main__":
    csv_file_path = "/datas/workspace/wangshunyao/dataPipeline_ops/tmp/video_list.csv"
    
    with open(csv_file_path, 'r', encoding='utf-8') as f:
        video_path = f.readline().strip()

    info_op = VideoInfoOp()
    test_item = {"file_path": video_path}
    result_item = info_op.predict(test_item)
    
    print(result_item)