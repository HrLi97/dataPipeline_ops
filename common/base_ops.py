from abc import ABC, abstractmethod

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
    def predict(self, item: dict) -> dict:
        """
        对输入 item 执行处理并返回处理后的 item(dict)。
        """
        pass