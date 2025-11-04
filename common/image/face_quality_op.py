"""
封装 FaceQualityModel)兼容版)
- 保留并把你原始导入放在文件最上方)如下所示,请勿改动这一行)
- 支持外部注入已初始化的 model)face_quality_model),或者传 model_path/clib_path 由算子延迟加载
- predict 接收 item['face_img'])BGR 或 RGB ndarray 或 PIL.Image),返回 item['face_quality'] = (score, msg)
- 若首次调用失败会尝试 RGB<->BGR 通道切换后重试一次
"""
import sys
import os
import logging
import numpy as np
from PIL import Image

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from dataPipeline_ops.third_part.CLIB_TensorRT.CLIB_FIQA.inference import FaceQualityModel
from ..base_ops import BaseOps
import cv2

class FaceQualityOp(BaseOps):
    def __init__(self, face_quality_model=None, model_path: str = None, clib_path: str = None, device: str = "cuda", **kwargs):
        """
        Args:
            face_quality_model: 已初始化的 FaceQualityModel 实例)优先使用)
            model_path/clib_path: 如果没有注入 model,则使用这些路径加载模型
            device: 传给 model)若需要),通常为 "cuda" 或 "cpu"
        """
        self._external_model = face_quality_model
        self.model_path = model_path
        self.clib_path = clib_path
        self.device = device

        # 实际使用的 model 引用)可能由外部注入,也可能由本类加载)
        self.model = face_quality_model if face_quality_model is not None else None
        self._loaded = face_quality_model is not None

    def _ensure_loaded(self):
        """如尚未加载则尝试加载模型)如果 FaceQualityModel 可用且 model_path/clib_path 提供)"""
        if self._loaded:
            return
        if FaceQualityModel is None:
            # 无法加载模型实现,记录警告
            logging.warning("FaceQualityOp: 找不到 FaceQualityModel 类)请检查 dataPipeline_ops.third_part 路径或 FIQA 模块)")
            self._loaded = False
            return
        try:
            # 若 model_path/clib_path 没有提供,尝试使用默认构造)有些实现允许 None)
            self.model = FaceQualityModel(self.model_path, self.clib_path) if (self.model_path or self.clib_path) else FaceQualityModel(None, None)
            self._loaded = True
        except Exception as e:
            logging.exception("FaceQualityOp: 加载 FaceQualityModel 失败: %s", e)
            self._loaded = False

    @staticmethod
    def _to_numpy_rgb(face_img):
        """
        将输入标准化为 RGB numpy ndarray)H,W,3),支持以下输入类型：
        - numpy ndarray (BGR 或 RGB)
        - PIL.Image
        尝试不破坏原始数组,仅在必要时做通道转换。
        """
        if face_img is None:
            return None
        # 如果是 PIL Image,转为 RGB ndarray
        if isinstance(face_img, Image.Image):
            arr = np.array(face_img.convert("RGB"))
            return arr
        # numpy array
        if isinstance(face_img, np.ndarray):
            if face_img.ndim == 2:
                # 灰度 -> 转 RGB
                return cv2.cvtColor(face_img, cv2.COLOR_GRAY2RGB)
            if face_img.ndim == 3 and face_img.shape[2] == 3:
                # 无法确定是 BGR 还是 RGB,这里不强制转换,返回原数组)上层尝试会做互转)
                return face_img
        # 兜底：尝试用 PIL 解析
        try:
            img = Image.fromarray(face_img)
            return np.array(img.convert("RGB"))
        except Exception:
            return None

    def predict(self, item: dict) -> dict:
        """
        item 输入：
            face_img: BGR 或 RGB ndarray,或 PIL.Image
        输出：
            item['face_quality'] = (score:float, msg:str)
            若出错,item['face_quality'] = (0.0, err_msg) 并在 item['face_quality_error'] 写错误详情
        """
        face = item.get("face_img")
        if face is None:
            item["face_quality"] = (0.0, "no_face_input")
            return item

        # 确保模型加载
        if not self._loaded:
            self._ensure_loaded()

        if self.model is None:
            item["face_quality"] = (0.0, "no_face_quality_model")
            item["face_quality_error"] = "FaceQualityModel 未加载或不可用"
            return item

        # 首次尝试直接传入)保持与原 model 接口兼容)
        try:
            # 某些实现可能要求 BGR 或 RGB; 我们先尝试以原始传入形式调用
            score_msg = self.model.predict_score(face)
            # 兼容返回 (score, msg) 或 dict 等情况
            if isinstance(score_msg, tuple) or isinstance(score_msg, list):
                score, msg = score_msg[0], score_msg[1] if len(score_msg) > 1 else ""
            elif isinstance(score_msg, dict):
                score = score_msg.get("score", 0.0)
                msg = score_msg.get("msg", "")
            else:
                # 如果返回单个标量
                try:
                    score = float(score_msg)
                    msg = ""
                except Exception:
                    score = 0.0
                    msg = str(score_msg)
            item["face_quality"] = (float(score), str(msg))
            return item
        except Exception as e_first:
            # 记录首次出错信息,继续尝试通道变换后再试一次
            logging.debug("FaceQualityOp: 首次 predict_score 失败,尝试 RGB<->BGR 转换后重试: %s", e_first)

        # 如果首次失败,尝试将输入转换为 RGB numpy,再做两次尝试：作为 RGB,或作为 BGR)互转)
        arr_rgb = self._to_numpy_rgb(face)
        if arr_rgb is None:
            item["face_quality"] = (0.0, "face_to_numpy_failed")
            item["face_quality_error"] = "无法将输入 face_img 转换为 numpy RGB"
            return item

        # 先当作 RGB 传入)有些模型期望 RGB)
        try:
            score_msg = self.model.predict_score(arr_rgb)
            if isinstance(score_msg, tuple) or isinstance(score_msg, list):
                score, msg = score_msg[0], score_msg[1] if len(score_msg) > 1 else ""
            elif isinstance(score_msg, dict):
                score, msg = score_msg.get("score", 0.0), score_msg.get("msg", "")
            else:
                try:
                    score = float(score_msg); msg = ""
                except Exception:
                    score, msg = 0.0, str(score_msg)
            item["face_quality"] = (float(score), str(msg))
            return item
        except Exception as e_rgb:
            logging.debug("FaceQualityOp: 作为 RGB 再次调用 predict_score 也失败: %s", e_rgb)

        # 再当作 BGR)把 RGB->BGR)重试
        try:
            arr_bgr = cv2.cvtColor(arr_rgb, cv2.COLOR_RGB2BGR)
            score_msg = self.model.predict_score(arr_bgr)
            if isinstance(score_msg, tuple) or isinstance(score_msg, list):
                score, msg = score_msg[0], score_msg[1] if len(score_msg) > 1 else ""
            elif isinstance(score_msg, dict):
                score, msg = score_msg.get("score", 0.0), score_msg.get("msg", "")
            else:
                try:
                    score = float(score_msg); msg = ""
                except Exception:
                    score, msg = 0.0, str(score_msg)
            item["face_quality"] = (float(score), str(msg))
            return item
        except Exception as e_bgr:
            logging.exception("FaceQualityOp: predict_score 多次尝试均失败: first=%s, rgb_try=%s, bgr_try=%s", e_first, e_rgb if 'e_rgb' in locals() else None, e_bgr)
            item["face_quality"] = (0.0, "predict_failed")
            item["face_quality_error"] = f"first_err={e_first}; rgb_err={e_rgb if 'e_rgb' in locals() else None}; bgr_err={e_bgr}"
            return item
