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
    def predict(self, *args: Any, **kwargs: Any):
        """
        对输入数据对象(或多个对象/参数)执行处理并返回同类型或相关的数据对象。
        """
        pass
    
    
'''
from abc import ABC, abstractmethod
from multimethod import multidispatch
from typing import overload, Iterable
from numbers import Real


class Interface(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def predict(self, *arg, **kwargs) -> int:
        pass


class MyClass(Interface):
    def __init__(self):
        super().__init__()

    @overload
    def predict(self, x: str, y: int = 1) -> int:
        """
        get int(x)*y
        """

    @overload
    def predict(self, x: float | int, y: int = 0) -> int:
        """
        get inx(x)+y
        """

    @overload
    def predict(self, x: Iterable[Real], y: int) -> float:
        """
        get sum(x)*y
        """
        pass

    @multidispatch
    def predict(self, *args) -> int:
        """
        test
        """
        raise NotImplementedError(tuple([type(v) for v in args]))

    @predict.register
    def _(self, x: str, y: int = 1) -> int:
        """
        test
        """
        return int(eval(x)) * y

    @predict.register
    def _(self, x: float | int, y: int) -> int:
        return int(x) + y

    @predict.register
    def _(self, x: Iterable[Real], y: int=1) -> float:
        return sum(x) * y

obj = MyClass()

print(obj.predict("5.1"))
print(obj.predict(x="2.1", y=5))
print(obj.predict(y=3, x=5.1))
print(obj.predict(1, 2))
print(obj.predict([1, 2, 3],2))
'''