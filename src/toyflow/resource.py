import asyncio
import logging
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List

logging.basicConfig(level=logging.INFO)


class ResourceType(Enum):
    CPU = "cpu"
    CUDA = "cuda"


@dataclass
class ResourceItem:
    rtype: ResourceType
    rid: int
    quantity: float = 1.0


class Resource(Dict[ResourceType, Dict[int, float]]):
    @classmethod
    def from_resource_items(cls, resource_items: List[ResourceItem]):
        result: Resource = cls(
            {rtype: defaultdict(float) for rtype in ResourceType}
        )
        for item in resource_items:
            result[item.rtype][item.rid] += item.quantity
        return result

    def get_cuda_ids(self):
        return list(self.get(ResourceType.CUDA, {}).keys())

    def as_dict(self):
        return {rtype.value: {rid: quantity for rid, quantity in resources.items()} for rtype, resources in self.items()}

    def add_(self, other: "Resource"):
        for rtype, resources in other.items():
            for rid, quantity in resources.items():
                self[rtype][rid] += quantity
        return self

    def minus_(self, other: "Resource"):
        for rtype, resources in other.items():
            for rid, quantity in resources.items():
                assert self[rtype][rid] >= quantity
                self[rtype][rid] -= quantity
        return self

    def split(self, requirement: Dict[ResourceType, List[float]]):
        try:
            allocated: Resource = Resource.from_resource_items([])
            all_resources = deepcopy(self)
            for rtype, quantities in requirement.items():
                if rtype not in all_resources:
                    raise ValueError(f"Resource type {rtype} not found")
                allocated[rtype] = self._greedy_allocate(
                    all_resources[rtype], quantities)
            return allocated
        except ValueError:
            return None

    @staticmethod
    def _greedy_allocate(available_resources: Dict[int, float], requests: List[float]) -> Dict[int, float]:
        available_resources = available_resources.copy()
        allocation = [None] * len(requests)
        # Sort requests from large to small
        sorted_requests = sorted(enumerate(requests), key=lambda x: -x[1])
        used_resources = set()
        for i, request in sorted_requests:
            # Sort available resources from small to large
            sorted_memories = sorted(
                available_resources.items(), key=lambda x: x[1])
            for rid, available_quantity in sorted_memories:
                if available_quantity >= request and rid not in used_resources:
                    available_resources[rid] -= request
                    allocation[i] = rid
                    used_resources.add(rid)
                    break
        if None in allocation:
            raise ValueError("Cannot allocate")
        return {allocation[i]: requests[i] for i in range(len(requests))}


class ResourcePool:
    def __init__(self, resource_items: List[ResourceItem]) -> None:
        self._resource: Resource = Resource.from_resource_items(resource_items)
        self.lock = asyncio.Lock()

    async def allocate_all(self):
        async with self.lock:
            result = self._resource
            self._resource = Resource.from_resource_items([])
            return result

    async def allocate(self, resource: Resource):
        async with self.lock:
            self._resource.minus_(resource)
            return resource

    async def release(self, resource: Resource):
        async with self.lock:
            self._resource.add_(resource)


if __name__ == '__main__':
    r = Resource()
    print(r.as_dict())
