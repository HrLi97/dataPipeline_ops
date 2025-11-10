"""
封装 RetinaFace 人脸检测：输入 frame(RGB)或 crop(BGR),返回 faces dict。
"""
from ..base_ops import BaseOps
from retinaface import RetinaFace 

class FaceDetectOp(BaseOps):
    def __init__(self, **kwargs):
        pass

    def predict(self, item: dict) -> dict:
        # 支持两种用法：若传入 "crop" 则在 crop 上检测；否则在 item["frame"] 全图检测并返回所有 faces
        crop = item.get("crop")  # BGR expected for RetinaFace.detect_faces in your code
        frame = item.get("frame")  # RGB expected -> convert before calling if needed
        try:
            if crop is not None:
                faces = RetinaFace.detect_faces(crop)
                item["faces_in_crop"] = faces
            elif frame is not None:
                import cv2
                bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                faces = RetinaFace.detect_faces(bgr)
                item["faces_in_frame"] = faces
            else:
                item["faces_in_frame"] = {}
        except Exception as e:
            item["face_detect_error"] = str(e)
            item.setdefault("faces_in_frame", {})
        return item

if __name__ == "__main__":
    import cv2

    csv_file_path = "/datas/workspace/wangshunyao/dataPipeline_ops/tmp/image_list.csv"
    
    with open(csv_file_path, 'r', encoding='utf-8') as f:
        image_path = f.readline().strip()

    # 算子需要的是图像数组(frame)，而不是路径，所以我们先用 cv2 读取
    # cv2.imread 默认读取为 BGR，算子的 "frame" 输入需要 RGB，所以转换一下
    bgr_image = cv2.imread(image_path)
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

    detector = FaceDetectOp()
    test_item = {"frame": rgb_image}
    result_item = detector.predict(test_item)
    
    # 为了不在终端打印巨大的图像数组，我们先将其从结果中移除
    if "frame" in result_item:
        del result_item["frame"]

    print(result_item)