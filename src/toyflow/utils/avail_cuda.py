import heapq
import subprocess


def avail_cuda_list(memory_requirement: int):
    with subprocess.Popen(
        "nvidia-smi -q -d Memory | grep -A5 GPU | grep Free", shell=True, stdout=subprocess.PIPE
    ) as p:
        free_mem = [(-int(x.split()[2]), i) for i, x in enumerate(p.stdout.readlines())]

    heapq.heapify(free_mem)

    def _get_one(free_mem):
        free, idx = free_mem[0]
        if free + memory_requirement > -20:
            return -1
        heapq.heapreplace(free_mem, (free + memory_requirement, idx))
        return idx

    result = []
    i = _get_one(free_mem)
    while i >= 0:
        result.append(i)
        i = _get_one(free_mem)
    return result
