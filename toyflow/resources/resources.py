import asyncio
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from typing import Any, Iterable, Optional, Union, List

__all__ = [
    "Resource",
    "DiscreteResource",
    "ContinuousResource",
    "CUDAResource",
    "MaxProcessesResource",
    "CPUResource",
]


class Resource(ABC):
    def __init__(self, resources: Any, name: str) -> None:
        super().__init__()
        self.name = name
        self.resources = resources
        self._lock = asyncio.Lock()

    @abstractmethod
    @asynccontextmanager
    async def allocate(self, quantity: Union[int, float, str]):
        yield


class DiscreteResource(Resource):
    def __init__(self, resources: Iterable[Any], name: str) -> None:
        _resources = asyncio.Queue()
        for item in resources:
            _resources.put_nowait(item)
        super().__init__(_resources, name)

    @asynccontextmanager
    async def allocate(self, quantity: int, timeout: Optional[float]):
        items = []
        timeout = timeout / quantity if (isinstance(timeout, float) and quantity > 0) else None
        try:
            for _ in range(quantity):
                item = await asyncio.wait_for(self.resources.get(), timeout=timeout)
                items.append(item)
            yield items
        except asyncio.TimeoutError:
            raise
        finally:
            for item in items:
                await self.resources.put(item)

    async def allocate_no_recycle(self, quantity, timeout: Optional[float]):
        timeout = timeout / quantity if (isinstance(timeout, float) and quantity > 0) else None
        async with self._lock:
            if (self.resources.qsize()) < quantity:
                return None
            items = []
            try:
                for _ in range(quantity):
                    item = await asyncio.wait_for(self.resources.get(), timeout=timeout)
                    items.append(item)
                return items
            except asyncio.TimeoutError:
                return None


class ContinuousResource(Resource):
    def __init__(self, resources: float, name: str) -> None:
        assert resources > 0
        super().__init__(resources, name)

    @asynccontextmanager
    async def allocate(self, quantity: float):
        async with asyncio.Lock():
            assert 0 <= quantity <= self.resources
            self.resources -= quantity
        yield quantity
        async with asyncio.Lock():
            self.resources += quantity


class CUDAResource(DiscreteResource):
    def __init__(self, device_list: Iterable[int], name: str = "cuda") -> None:
        """
        device_list means the device ids.
        """
        super().__init__(device_list, name)


class MaxProcessesResource(ContinuousResource):
    def __init__(self, max_process: Optional[int] = None, name: str = "max_processes") -> None:
        if max_process is None:
            _max_process = float("inf")
        else:
            _max_process = float(max_process)
        super().__init__(float(_max_process), name)


class CPUResource(ContinuousResource):
    """
    Total resource is 1, and each allocation quantity should be in [0, 1].
    """

    def __init__(self, name: str = "cpu") -> None:
        super().__init__(resources=1, name=name)
