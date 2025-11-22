from abc import ABC, abstractmethod
from typing import Any

class BaseOps(ABC):
    """
    算子基类（抽象基类）。
    所有具体算子必须继承此类并实现 __init__ 和 predict 方法。
    输入输出采用 dict(item),便于 pipeline 串联。
    """

    @abstractmethod
    def __init__(self, **kwargs):
        """
        初始化算子所需参数。
        """
        pass

    @abstractmethod
    def predict(self, data: Any):
        """
        对输入数据对象执行处理并返回同类型或相关的数据对象。
        """
        pass