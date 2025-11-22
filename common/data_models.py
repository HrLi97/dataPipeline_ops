from typing import Any, Optional, List, Dict
import torch
import numpy as np
import ffmpeg

class BaseData:
    def __init__(self, tensor: Optional[torch.Tensor] = None, format: str = "NCHW", scale: Optional[float] = None):
        self.data_ = tensor
        self.format_ = format.upper()
        self.scale = scale

    def clone(self):
        t = self.data_.clone() if isinstance(self.data_, torch.Tensor) else self.data_
        return self.__class__(tensor=t, format=self.format_, scale=self.scale)

    def values_like(self, value: Optional[float] = None):
        new = self.__class__(tensor=torch.empty_like(self.data_) if isinstance(self.data_, torch.Tensor) else None, format=self.format_, scale=self.scale)
        if value is not None and isinstance(new.data_, torch.Tensor):
            new.data_.fill_(value)
        return new

    def set_scale(self, scale: float):
        self.scale = scale

    def is_norm(self) -> bool:
        if not isinstance(self.data_, torch.Tensor):
            return False
        return self.data_.dtype not in (torch.uint8, torch.uint16, torch.uint32, torch.uint64)

    def try_norm(self, dtype: torch.dtype = torch.float32, allow_deduce: bool = True) -> bool:
        if not isinstance(self.data_, torch.Tensor):
            return False
        if self.is_norm():
            self.data_ = self.data_.to(dtype)
            return True
        s = self.scale
        if s is None and allow_deduce:
            dt = self.data_.dtype
            if dt == torch.uint8:
                s = 255.0
            elif dt == torch.uint16:
                s = 65535.0
        if s is None:
            return False
        self.scale = s
        self.data_ = (self.data_.to(torch.float32) / s).to(dtype)
        return True

    def try_denorm(self, img_type: Optional[torch.dtype] = None, scale: Optional[float] = None) -> bool:
        if not isinstance(self.data_, torch.Tensor):
            return False
        if not self.is_norm():
            return True
        s = scale if scale is not None else self.scale
        if s is None:
            return False
        x = self.data_.to(torch.float32) * s
        if img_type is None:
            img_type = torch.uint8 if s <= 255 else torch.uint16
        self.data_ = x.to(img_type)
        return True

    def to_device(self, device: Any):
        if isinstance(self.data_, torch.Tensor):
            self.data_ = self.data_.to(device)
        return self

    def to_format(self, format: str = "NCHW"):
        f = format.upper()
        if f == "NCHW":
            return self.to_nchw()
        else:
            return self.to_nhwc()

    def to_nchw(self):
        if isinstance(self.data_, torch.Tensor) and self.format_ != "NCHW":
            if self.data_.ndim == 4:
                self.data_ = self.data_.contiguous().permute(0, 3, 1, 2)
            elif self.data_.ndim == 3:
                self.data_ = self.data_.contiguous().permute(2, 0, 1)
            self.format_ = "NCHW"
        return self

    def to_nhwc(self):
        if isinstance(self.data_, torch.Tensor) and self.format_ != "NHWC":
            if self.data_.ndim == 4:
                self.data_ = self.data_.contiguous().permute(0, 2, 3, 1)
            elif self.data_.ndim == 3:
                self.data_ = self.data_.contiguous().permute(1, 2, 0)
            self.format_ = "NHWC"
        return self

    @property
    def data(self):
        return self.data_

    @data.setter
    def data(self, value):
        self.data_ = value

    @property
    def format(self):
        return self.format_

    @property
    def device(self):
        return self.data_.device if isinstance(self.data_, torch.Tensor) else None

    @property
    def dtype(self):
        return self.data_.dtype if isinstance(self.data_, torch.Tensor) else None

    @property
    def shape(self):
        return list(self.data_.shape) if isinstance(self.data_, torch.Tensor) else None

    @property
    def num_frame(self):
        s = self.shape
        if not s:
            return None
        return s[0] if self.format_ == "NCHW" else s[0]

    @property
    def channel(self):
        s = self.shape
        if not s:
            return None
        return s[1] if self.format_ == "NCHW" else s[3] if len(s) >= 4 else (s[2] if len(s) == 3 else None)

    @property
    def height(self):
        s = self.shape
        if not s:
            return None
        return s[2] if self.format_ == "NCHW" else s[1] if len(s) >= 3 else None

    @property
    def width(self):
        s = self.shape
        if not s:
            return None
        return s[3] if self.format_ == "NCHW" else s[2] if len(s) >= 3 else None

    def to_item(self, item: Optional[dict] = None) -> dict:
        meta = {"format": self.format_, "shape": self.shape, "scale": self.scale}
        return {f"{self.__class__.__name__.lower()}_meta": meta}

class VideoData(BaseData):
    def __init__(self, tensor: Optional[torch.Tensor] = None, format: str = "NCHW", scale: Optional[float] = None):
        super().__init__(tensor=tensor, format=format, scale=scale)

    def clone(self) -> "VideoData":
        t = self.data_.clone() if isinstance(self.data_, torch.Tensor) else self.data_
        return VideoData(tensor=t, format=self.format_, scale=self.scale)

    def to_item(self, item: Optional[dict] = None) -> dict:
        return super().to_item(item)


    def is_norm(self) -> bool:
        return super().is_norm()

    def try_norm(self, dtype: torch.dtype = torch.float32, allow_deduce: bool = True) -> bool:
        return super().try_norm(dtype=dtype, allow_deduce=allow_deduce)

    def try_denorm(self, img_type: Optional[torch.dtype] = None, scale: Optional[float] = None) -> bool:
        return super().try_denorm(img_type=img_type, scale=scale)

    def to_device(self, device: Any) -> "VideoData":
        super().to_device(device)
        return self

    def to_format(self, format: str = "NCHW") -> "VideoData":
        super().to_format(format)
        return self

    def to_nchw(self) -> "VideoData":
        super().to_nchw()
        return self

    def to_nhwc(self) -> "VideoData":
        super().to_nhwc()
        return self

    @property
    def data(self):
        return super().data

    @data.setter
    def data(self, value):
        self.data_ = value

    @property
    def format(self):
        return super().format

    @property
    def shape(self):
        return super().shape

    @property
    def num_frame(self):
        return super().num_frame

    @property
    def channel(self):
        return super().channel

    @property
    def height(self):
        return super().height

    @property
    def width(self):
        return super().width

class ImageData(BaseData):
    def __init__(self, tensor: Optional[torch.Tensor] = None, format: str = "NCHW", scale: Optional[float] = None):
        super().__init__(tensor=tensor, format=format, scale=scale)

    def to_item(self, item: Optional[dict] = None) -> dict:
        return super().to_item(item)

    @classmethod
    def from_path(cls, path: str, to_format: str = "NHWC", scale: float = 255.0):
        import cv2, numpy as np, torch
        bgr = cv2.imread(path)
        if bgr is None:
            raise FileNotFoundError(f"failed to read image: {path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        t = torch.from_numpy(rgb)
        if to_format.upper() == "NCHW":
            t = t.permute(2, 0, 1)
        return cls(tensor=t, format=to_format.upper(), scale=scale)


class TextData(BaseData):
    def __init__(self, tensor: Optional[torch.Tensor] = None, format: str = "NT", scale: Optional[float] = None):
        super().__init__(tensor=tensor, format=format, scale=scale)

    def to_item(self, item: Optional[dict] = None) -> dict:
        return super().to_item(item)


class AudioData(BaseData):
    def __init__(self, tensor: Optional[torch.Tensor] = None, format: str = "NCL", scale: Optional[float] = None):
        super().__init__(tensor=tensor, format=format, scale=scale)

    def to_item(self, item: Optional[dict] = None) -> dict:
        return super().to_item(item)

class FFmpegVideoInfo:
    def __init__(self, file_path: Optional[str] = None):
        self.file_path = file_path
        self.num_frame = 0
        self.channel = 0
        self.width = 0
        self.height = 0
        self.fps = 0.0
        self.bit_rate = 0
        self.pix_fmt = None
        self.duration = 0.0

    def init(self, file_path: Optional[str] = None) -> bool:
        if file_path is not None:
            self.file_path = file_path
        try:
            meta = ffmpeg.probe(self.file_path)
            vs = next(s for s in meta["streams"] if s.get("codec_type") == "video")
            self.width = int(vs.get("width") or 0)
            self.height = int(vs.get("height") or 0)
            fr = vs.get("avg_frame_rate") or "0/1"
            a, b = fr.split("/")
            self.fps = float(a) / float(b) if float(b) != 0 else 0.0
            self.pix_fmt = vs.get("pix_fmt")
            self.bit_rate = int(vs.get("bit_rate") or meta.get("format", {}).get("bit_rate") or 0)
            dur = meta.get("format", {}).get("duration")
            self.duration = float(dur) if dur is not None else 0.0
            nf = vs.get("nb_frames")
            if nf is not None:
                self.num_frame = int(nf)
            else:
                self.num_frame = int(round(self.duration * self.fps)) if self.fps > 0 and self.duration > 0 else 0
            self.channel = 3
            return self.num_frame > 0 and self.width > 0 and self.height > 0
        except Exception:
            return False

    def read_device(self, device_str: str) -> tuple:
        if ":" in device_str:
            d, i = device_str.split(":")
            return d.strip().lower(), int(i.strip())
        return device_str.strip().lower(), 0

    def load_data_by_index(self, indexs: List[int], device: str = "cpu", format: str = "NHWC") -> VideoData:
        if not indexs:
            return VideoData(tensor=None, format=format)
        expr = "+".join([f"eq(n\\,{i})" for i in indexs])
        out = ffmpeg.input(self.file_path).filter("select", expr).output(
            "pipe:", format="rawvideo", pix_fmt="rgb24", vframes=len(indexs)
        )
        raw, _ = ffmpeg.run(out, capture_stdout=True, capture_stderr=True)
        h, w = self.height, self.width
        arr = np.frombuffer(raw, np.uint8)
        if arr.size != len(indexs) * h * w * 3:
            return VideoData(tensor=None, format=format)
        arr = arr.reshape((len(indexs), h, w, 3))
        ten = torch.from_numpy(arr)
        dev, _ = self.read_device(device)
        if dev != "cpu":
            ten = ten.to(dev)
        vd = VideoData(tensor=ten, format="NHWC", scale=255.0)
        if format.upper() == "NCHW":
            vd.to_format("NCHW")
        return vd

    def load_data(self, offset: int, max_num_frame: int, sample_rate: int = 1) -> VideoData:
        if max_num_frame < 0:
            max_num_frame = self.num_frame
        idxs = list(range(offset, min(offset + sample_rate * max_num_frame, self.num_frame), sample_rate))[:max_num_frame]
        return self.load_data_by_index(idxs)