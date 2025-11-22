import torch
import numpy as np
import copy
from abc import ABC, abstractmethod
from typing import Any, Optional, List, Dict, Union, Tuple
from pathlib import Path

# --- 依赖检查与导入 ---
try:
    import decord # pip install decord (强烈推荐用于视频读取)
    decord.bridge.set_bridge('torch')
except ImportError:
    decord = None

try:
    import cv2
except ImportError:
    cv2 = None

# ==========================================
# 1. 抽象基类 (BaseData)
# ==========================================

class BaseData(ABC):
    def __init__(self, tensor: Optional[torch.Tensor] = None, format: str = "NCHW", scale: Optional[float] = None, meta: Dict = None):
        """
        tensor: 核心数据容器
        format: NCHW, NHWC, NTCHW (Video), NTHWC (Video)
        scale: 归一化系数 (如 255.0)
        meta: 存储元数据 (如 fps, 原始路径, 文本内容等)
        """
        self.data_ = tensor
        self.format_ = format.upper()
        self.scale = scale
        self.meta = meta if meta else {}

    def clone(self):
        """深拷贝"""
        t = self.data_.clone() if isinstance(self.data_, torch.Tensor) else copy.deepcopy(self.data_)
        return self.__class__(tensor=t, format=self.format_, scale=self.scale, meta=copy.deepcopy(self.meta))

    def to_device(self, device: Any):
        """统一设备迁移"""
        if isinstance(self.data_, torch.Tensor):
            self.data_ = self.data_.to(device)
        return self

    # --- 核心：归一化逻辑 ---
    
    def is_norm(self) -> bool:
        if not isinstance(self.data_, torch.Tensor):
            return False
        # 浮点型通常意味着已归一化，整型意味着未归一化
        return self.data_.dtype not in (torch.uint8, torch.uint16, torch.uint32, torch.uint64)

    def try_norm(self, dtype: torch.dtype = torch.float32, allow_deduce: bool = True) -> bool:
        """尝试将 uint8/16 转为 float32 并归一化到 [0, 1]"""
        if not isinstance(self.data_, torch.Tensor):
            return False
        
        if self.is_norm():
            if self.data_.dtype != dtype:
                self.data_ = self.data_.to(dtype)
            return True

        # 自动推断 Scale
        s = self.scale
        if s is None and allow_deduce:
            dt = self.data_.dtype
            if dt == torch.uint8:
                s = 255.0
            elif dt == torch.uint16:
                s = 65535.0
        
        if s is None:
            return False # 无法推断，放弃归一化

        self.scale = s
        self.data_ = (self.data_.to(torch.float32) / s).to(dtype)
        return True

    def try_denorm(self, img_type: Optional[torch.dtype] = None, scale: Optional[float] = None) -> bool:
        """尝试将 float32 [0, 1] 还原为 uint8 [0, 255]"""
        if not isinstance(self.data_, torch.Tensor):
            return False
        if not self.is_norm():
            return True

        s = scale if scale is not None else self.scale
        if s is None:
            # 默认假设
            s = 255.0 
        
        x = self.data_.to(torch.float32) * s
        
        if img_type is None:
            if s <= 255: img_type = torch.uint8
            elif s <= 65535: img_type = torch.uint16
            else: img_type = torch.int32
            
        self.data_ = x.to(img_type)
        return True

    # --- 格式转换逻辑 (Format Conversion) ---

    def to_format(self, target_format: str):
        target_format = target_format.upper()
        if target_format == self.format_:
            return self
            
        # 简单处理 NCHW <-> NHWC 的互转
        if "H" in target_format and "W" in target_format and "C" in target_format:
             if target_format.endswith("C") and not self.format_.endswith("C"):
                 return self.to_nhwc()
             elif not target_format.endswith("C") and self.format_.endswith("C"):
                 return self.to_nchw()
        
        # 若涉及更复杂的维度变换（如 Video T 维），可在子类覆盖或此处扩展
        self.format_ = target_format
        return self

    def to_nchw(self):
        if not isinstance(self.data_, torch.Tensor) or self.format_.endswith("HW"): # 已经是 *HW
            return self
            
        # 处理 4D (Image)
        if self.data_.ndim == 4: 
            self.data_ = self.data_.permute(0, 3, 1, 2).contiguous() # NHWC -> NCHW
            self.format_ = self.format_.replace("NHWC", "NCHW")
        # 处理 5D (Video)
        elif self.data_.ndim == 5:
            self.data_ = self.data_.permute(0, 1, 4, 2, 3).contiguous() # NTHWC -> NTCHW
            self.format_ = "NTCHW"
            
        return self

    def to_nhwc(self):
        if not isinstance(self.data_, torch.Tensor) or self.format_.endswith("WC"):
            return self

        # 处理 4D (Image)
        if self.data_.ndim == 4:
            self.data_ = self.data_.permute(0, 2, 3, 1).contiguous() # NCHW -> NHWC
            self.format_ = "NHWC"
        # 处理 5D (Video)
        elif self.data_.ndim == 5:
            self.data_ = self.data_.permute(0, 1, 3, 4, 2).contiguous() # NTCHW -> NTHWC
            self.format_ = "NTHWC"
            
        return self

    # --- 智能属性 (Smart Properties) ---
    
    @property
    def device(self):
        return self.data_.device if isinstance(self.data_, torch.Tensor) else None

    @property
    def shape(self):
        return list(self.data_.shape) if isinstance(self.data_, torch.Tensor) else []

    @property
    def channel(self):
        if not self.shape: return None
        try:
            return self.shape[self.format_.find('C')]
        except ValueError: return None # Format string doesn't match

    @property
    def height(self):
        if not self.shape: return None
        try:
            return self.shape[self.format_.find('H')]
        except ValueError: return None

    @property
    def width(self):
        if not self.shape: return None
        try:
            return self.shape[self.format_.find('W')]
        except ValueError: return None
    
    @abstractmethod
    def show(self):
        """可视化调试"""
        pass

# ==========================================
# 2. 视频容器 (VideoData)
# ==========================================

class VideoData(BaseData):
    def __init__(self, source: Union[str, torch.Tensor, 'VideoData'] = None, format="NTCHW", scale=None):
        """
        source: 视频路径(str), Tensor, 或者另一个VideoData
        """
        super().__init__(format=format, scale=scale)
        
        # 1. 从路径加载
        if isinstance(source, (str, Path)):
            self.meta['source_path'] = str(source)
            self._load_from_path(str(source)) # 内部会设置 self.data_
        
        # 2. 从 Tensor 加载
        elif isinstance(source, torch.Tensor):
            self.data_ = source
            # 自动补齐维度: 假设输入是 TCHW -> NTCHW
            if self.data_.ndim == 4:
                self.data_ = self.data_.unsqueeze(0)
        
        # 3. Copy 构造
        elif isinstance(source, VideoData):
            self.data_ = source.data_.clone()
            self.format_ = source.format_
            self.scale = source.scale
            self.meta = copy.deepcopy(source.meta)

    def _load_from_path(self, path: str):
        """使用 Decord 高效加载"""
        if decord is None:
            raise ImportError("Please install decord: pip install decord")
        
        vr = decord.VideoReader(path)
        self.meta['fps'] = vr.get_avg_fps()
        self.meta['total_frames'] = len(vr)
        self.meta['duration'] = len(vr) / vr.get_avg_fps() if vr.get_avg_fps() > 0 else 0
        
        # 默认行为：为了演示，加载前 16 帧。
        # 实际 Pipeline 中，建议实现 Lazy Loading，这里只存路径，crop时再读
        max_frames = min(16, len(vr))
        indices = list(range(max_frames))
        
        # decord 返回 (T, H, W, C)
        frames = vr.get_batch(indices) 
        
        # 转为 Tensor: (1, T, C, H, W)
        # frames 是 decord 的 Bridge Tensor，已经是 Tensor 了
        self.data_ = frames.float().permute(0, 3, 1, 2).unsqueeze(0)
        self.format_ = "NTCHW" 
        
        # 自动归一化处理
        self.scale = 255.0
        self.data_ = self.data_ / self.scale

    @property
    def num_frame(self):
        if not self.shape: return 0
        idx = self.format_.find('T')
        return self.shape[idx] if idx != -1 else 1

    @property
    def fps(self):
        return self.meta.get('fps', 30.0)

    def get_frame(self, index: int) -> "ImageData":
        """提取单帧为 ImageData"""
        # 假设 NTCHW
        if "T" in self.format_:
             t_idx = self.format_.find('T')
             # 选取特定帧，保持维度 NCHW
             frame_data = self.data_.select(t_idx, index)
             # 移除 T 后，格式变为 NCHW
             new_fmt = self.format_.replace("T", "")
             
             img = ImageData(frame_data, format=new_fmt, scale=self.scale)
             img.meta = copy.deepcopy(self.meta)
             img.meta['frame_index'] = index
             return img
        return None

    def crop(self, bbox: List[int]):
        """裁剪 [x, y, w, h]"""
        x, y, w, h = bbox
        if w < 0: w = self.width
        if h < 0: h = self.height
        
        # 利用 BaseData 的属性直接定位 H 和 W，不用管格式
        h_idx = self.format_.find('H')
        w_idx = self.format_.find('W')
        
        # 使用 torch.narrow 进行切片
        self.data_ = self.data_.narrow(h_idx, y, h).narrow(w_idx, x, w)
        return self

    def show(self):
        print(f"[VideoData] Shape:{self.shape} Format:{self.format_} FPS:{self.fps:.2f} Norm:{self.is_norm()}")

# ==========================================
# 3. 图像容器 (ImageData)
# ==========================================

class ImageData(BaseData):
    def __init__(self, source: Union[str, np.ndarray, torch.Tensor] = None, format="NCHW", scale=None):
        super().__init__(format=format, scale=scale)
        
        # 1. 路径
        if isinstance(source, (str, Path)):
            self.meta['source_path'] = str(source)
            self._load_from_path(str(source))
            
        # 2. Numpy (通常来自 cv2/PIL)
        elif isinstance(source, np.ndarray):
            self.data_ = torch.from_numpy(source)
            # HWC -> NCHW
            if self.data_.ndim == 3:
                self.data_ = self.data_.permute(2, 0, 1).unsqueeze(0)
            self.format_ = "NCHW"
            self.try_norm()

        # 3. Tensor
        elif isinstance(source, torch.Tensor):
            self.data_ = source
            if self.data_.ndim == 3:
                self.data_ = self.data_.unsqueeze(0)
    
    def _load_from_path(self, path: str):
        if cv2 is None:
            raise ImportError("opencv-python is required")
        
        bgr = cv2.imread(path)
        if bgr is None:
            raise FileNotFoundError(f"Cannot read image: {path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        
        # 统一转为 NCHW, float32, 0-1
        self.data_ = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        self.format_ = "NCHW"
        self.scale = 255.0

    def resize(self, size: Tuple[int, int]):
        """size: (H, W)"""
        # 确保 NCHW 才能用 interpolate
        prev_fmt = self.format_
        self.to_nchw()
        self.data_ = torch.nn.functional.interpolate(
            self.data_, size=size, mode='bilinear', align_corners=False
        )
        self.to_format(prev_fmt) # 恢复原格式
        return self

    def show(self):
        print(f"[ImageData] Shape:{self.shape} Format:{self.format_} Scale:{self.scale}")

# ==========================================
# 4. 文本容器 (TextData)
# ==========================================

class TextData(BaseData):
    def __init__(self, source: Union[str, List[int], torch.Tensor] = None):
        super().__init__(format="NT") # N: Batch, T: Sequence Length
        self.raw_text = None
        
        if isinstance(source, str):
            self.raw_text = source
            self.data_ = None # 此时还没有 Tensor，直到 tokenize
        elif isinstance(source, torch.Tensor):
            self.data_ = source
            
    def tokenize(self, tokenizer, max_length=128):
        if self.raw_text:
            # 伪代码：调用 huggingface tokenizer
            # out = tokenizer(self.raw_text, ...)
            # self.data_ = out['input_ids']
            pass
        return self
        
    def show(self):
        content = self.raw_text if self.raw_text else f"Tensor shape {self.shape}"
        print(f"[TextData] {content}")

# ==========================================
# 使用演示 (Pipeline Demo)
# ==========================================

if __name__ == "__main__":
    # 模拟创建一个虚假视频文件，避免运行报错
    # 在实际使用中，你直接传真实路径即可
    dummy_video_path = "test_video.mp4" 
    
    print("=== 1. 统一初始化测试 ===")
    # 假设我们有一个 Tensor 模拟视频输入 (T=5, C=3, H=64, W=64)
    raw_video_tensor = torch.randint(0, 255, (5, 3, 64, 64), dtype=torch.uint8)
    
    # 直接初始化，自动识别 Tensor，自动归一化，自动 unsqueeze Batch
    video = VideoData(raw_video_tensor, format="TCHW") 
    print(f"Video Init: {video.shape} | Is Norm: {video.is_norm()}") # 应为 [1, 5, 3, 64, 64]

    print("\n=== 2. 格式转换与操作测试 ===")
    # 转换格式 NCHW <-> NHWC
    video.to_nhwc()
    print(f"To NTHWC: {video.shape} | Format tag: {video.format_}")
    
    # 裁剪 (利用 BaseData 的宽属性)
    video.crop([0, 0, 32, 32])
    print(f"Cropped:  {video.shape}")

    print("\n=== 3. 跨模态转换测试 (Video -> Image) ===")
    video.to_nchw() # 转回方便处理
    frame_0 = video.get_frame(0) # 提取第0帧
    print(f"Frame 0 extracted: Type={type(frame_0)} | Shape={frame_0.shape}")
    
    # 对提取出的 Image 做操作
    frame_0.resize((128, 128))
    print(f"Frame 0 resized:   {frame_0.shape}")

    print("\n=== 4. 模拟文件加载 (需要 decord/cv2) ===")
    try:
        # img = ImageData("my_image.jpg")
        # vid = VideoData("my_video.mp4")
        print("Dependencies installed, file loading available.")
    except ImportError as e:
        print(f"Skipping file load test: {e}")