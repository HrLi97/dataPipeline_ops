"""
封装 RetinaFace 人脸检测：输入 frame(RGB)或 crop(BGR),返回 faces dict。
"""
from ..base_ops import BaseOps
from ..data_models import ImageData,FFmpegVideoInfo
from retinaface import RetinaFace 
import cv2
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

class FaceDetectOp(BaseOps):
    def __init__(self, **kwargs):
        pass

    def predict(self, data: ImageData) -> ImageData:
        x = data.data
        # try:
        if x is None:
            return data
        # 确保是 NHWC 并在 CPU
        if data.format.upper() == "NCHW":
            x = x.contiguous().permute(0, 2, 3, 1) if x.ndim == 4 else x.contiguous().permute(1, 2, 0)
        # 取第一帧或单张
        if x.ndim == 4:
            x = x[0]
        # 反归一化到 uint8
        if data.is_norm():
            s = data.scale if data.scale is not None else 255.0
            x = (x.to(dtype=x.dtype) * s).clamp(0, 255)
        x_np = x.detach().cpu().numpy().astype(np.uint8)
        # RGB -> BGR
        if x_np.ndim == 3 and x_np.shape[2] == 3:
            x_bgr = cv2.cvtColor(x_np, cv2.COLOR_RGB2BGR)
        else:
            x_bgr = x_np
        faces = RetinaFace.detect_faces(x_bgr)
        setattr(data, "faces", faces)
        # except Exception:
        #     setattr(data, "faces", {})
        return data

if __name__ == "__main__":
    import torch
    
    ffmpegVideoInfo = FFmpegVideoInfo("/mnt/cfs/shanhai/lihaoran/project/code/color/data/content/video/城市4k.mp4")
    init = ffmpegVideoInfo.init()
    print(ffmpegVideoInfo.width)
    print(ffmpegVideoInfo.bit_rate)
    csv_file_path = "/mnt/cfs/shanhai/lihaoran/project/code/dataPipeline_ops/tmp/image_list.csv"
    with open(csv_file_path, 'r', encoding='utf-8') as f:
        image_path = f.readline().strip()
    img = ImageData.from_path(image_path, to_format="NHWC", scale=255.0)
    detector = FaceDetectOp()x
    out = detector.predict(img)
    print("faces keys:", list(getattr(out, "faces", {}).keys()))