from abc import ABC, abstractmethod
from typing import Any, Dict

class BaseOps(ABC):
    """
    算子基类。
    支持灵活的 predict 参数定义，同时保留作为 Pipeline 节点的统一调用能力。
    """

    def __init__(self):
        pass

    @abstractmethod
    def predict(self, *args, **kwargs) -> Any:
        """
        核心预测逻辑。
        子类应重写此方法，并可以定义具体的参数，例如：
        def predict(self, video: VideoData) -> List[ImageData]: ...
        """
        pass

    def __call__(self, *args, **kwargs) -> Any:
        """
        允许像函数一样调用算子：op(data) 相当于 op.predict(data)
        """
        return self.predict(*args, **kwargs)