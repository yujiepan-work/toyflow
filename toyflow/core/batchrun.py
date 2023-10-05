import asyncio
import heapq
import json
import logging
import os
import platform
import subprocess
import time
from contextlib import asynccontextmanager, nullcontext
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union
from unittest.mock import Mock

from toyflow.core.task_node import Task
from toyflow.dag import TopoSorter
from toyflow.resources import CPUResource, CUDAResource
from toyflow.utils import logger

CUDA_VISIBLE_DEVICES = "CUDA_VISIBLE_DEVICES"


def pre_launch_worker(io_folder):
    Path(io_folder).mkdir()


def task_sort_key(task: Task):
    return (task.cuda_quantity, task.cpu_quantity)


from dataclasses import dataclass, field
from typing import Any, Union, List, Tuple


@dataclass(order=True)
class PrioritizedTask:
    priority: tuple[Union[int, float], ...]
    task: Task = field(compare=False)

    @classmethod
    def from_task(cls, task: Task):
        return cls(priority=(task.cuda_quantity, task.cpu_quantity), task=task)


class Launcher:
    def __init__(self, cuda_list: List[int]) -> None:
        self.cuda_list = cuda_list
        self.add_timestamp_to_log = False

    def run(self, tasks: List[Task], add_timestamp_to_log=False):
        self.add_timestamp_to_log = add_timestamp_to_log
        runner = asyncio.run(self._run(tasks))
        print(runner.get_current_ordered_nodes())

    async def _update_candidates(self, finished_task: Task, runner: TopoSorter, cuda_resource: CUDAResource):
        new_nodes = runner.set_next_node(finished_task)
        async with asyncio.Lock():
            for node in new_nodes:
                await self.candidates.put(PrioritizedTask.from_task(node))

    async def _run(self, tasks):
        cuda_resource = CUDAResource(self.cuda_list)
        cpu_resource = CPUResource()
        runner = TopoSorter.from_nodes(tasks)

        self.launched_tasks = list()
        candidates = runner.get_next_node_candidates()
        self.candidates = asyncio.PriorityQueue()
        for c in candidates:
            self.candidates.put_nowait(PrioritizedTask.from_task(c))

        self.num_launched_tasks = 0
        while self.num_launched_tasks < len(tasks):
            try:
                next_task: PrioritizedTask = await asyncio.wait_for(self.candidates.get(), timeout=1)
                logger.critical(f"next_task poll: {next_task.task.identifier}")
                async_task = asyncio.create_task(
                    self._launch_task_and_then_update(
                        next_task.task,
                        len(self.launched_tasks) + 1,
                        cuda_resource,
                        runner,
                    )
                )
                async_task.add_done_callback(self.launched_tasks.remove)
                self.launched_tasks.append(async_task)
            except asyncio.TimeoutError:
                pass
            await asyncio.sleep(0.01)

        if self.launched_tasks:
            await asyncio.wait(self.launched_tasks)
        return runner

    async def _launch_task_and_then_update(
        self, task: Task, task_id: int, resource_manager: CUDAResource, runner: TopoSorter, **kwargs
    ):
        try:
            async with resource_manager.allocate(quantity=task.cuda_quantity, timeout=0.2) as cuda_list:
                self.num_launched_tasks += 1
                await self._launch_task(task, task_id, resource_manager, cuda_list=cuda_list, **kwargs)
                await self._update_candidates(task, runner, resource_manager)
        except asyncio.TimeoutError:
            await self.candidates.put(PrioritizedTask.from_task(task))
        finally:
            pass

    async def _launch_task(self, task: Task, task_id: int, resource_manager: CUDAResource, **kwargs) -> str:
        total = str(kwargs.get("total", "?"))
        cuda_list = kwargs.get("cuda_list", tuple())
        # async with resource_manager.allocate(quantity=task.cuda_quantity, timeout=1) as cuda_list:
        with nullcontext():
            cuda = ",".join(map(str, cuda_list))
            logging_callback = lambda pid: logger.info(
                "Running Task[%d/%s] PID=%d CUDA=%s: %s", task_id, total, pid, cuda, task.identifier
            )
            env = deepcopy(task.env)
            env[CUDA_VISIBLE_DEVICES] = str(cuda)
            task.prepare_fn(*task.prepare_fn_args)

            io_folder = Path(task.io_folder)
            io_folder.mkdir(exist_ok=True, parents=True)
            task_description_folder = io_folder / "task_information"
            task_description_folder.mkdir(exist_ok=True, parents=True)
            full_info = {
                "cmd_str": task.cmd_str(),
                "cwd": Path(task.cwd).absolute().as_posix(),
                "cmd_list": task.cmd_list(),
                "env": dict(sorted(env.items())),
                "host": platform.uname()._asdict(),
                "launch_time": time.localtime(),
            }
            with open(task_description_folder / "full_description.json", "w", encoding="utf-8") as f_task_desc:
                json.dump(
                    full_info,
                    f_task_desc,
                    indent=4,
                )
            with open(task_description_folder / "task_script.bash", "w", encoding="utf-8") as f_task_desc:
                f_task_desc.write(task.cmd_bash())

            with open(io_folder / "task_description.json", "w", encoding="utf-8") as f_task_desc:
                cwd = Path(task.cwd).absolute()
                json.dump(
                    {
                        "cwd": cwd.relative_to(Path.home()).as_posix()
                        if cwd.is_relative_to(Path.home())
                        else cwd.as_posix(),
                        "cmd_list": task.cmd_list(),
                    },
                    f_task_desc,
                    indent=4,
                )
            start_time = time.time()
            proc = await self._run_single_process(
                task.cmd_str(), task.io_folder, task.cwd, env, task_description_folder, full_info, logging_callback
            )
            status = "SUCCESS" if proc.returncode == 0 else "FAIL"
            cost_time = time.time() - start_time
            log_fn = logger.warning if proc.returncode == 0 else logger.error
            log_fn(
                "%s Task[%d/%s] PID=%d CUDA=%s (time: %ds): %s",
                status,
                task_id,
                total,
                proc.pid,
                cuda,
                int(cost_time),
                task.identifier,
            )
            if cost_time < 30:
                with open(io_folder / "END_QUICKLY", "w", encoding="utf-8") as f:
                    f.write(f"cost_time: {cost_time} seconds.")

            full_info["end_time"] = time.localtime()
            with open(task_description_folder / "full_description.json", "w", encoding="utf-8") as f_task_desc:
                json.dump(
                    full_info,
                    f_task_desc,
                    indent=4,
                )
            return status

    async def _run_single_process(
        self,
        cmd: str,
        io_folder: Union[Path, str],
        cwd: str,
        env: Dict[str, str],
        task_description_folder,
        full_info,
        logging_callback,
    ):
        io_folder = Path(io_folder)
        timestamp = ("_" + str(int(time.time()))) if self.add_timestamp_to_log else ""
        with open(io_folder / f"stdout{timestamp}.log", "w", encoding="utf-8") as f_out, open(
            io_folder / f"stderr{timestamp}.log", "w", encoding="utf-8"
        ) as f_err:
            proc = await asyncio.create_subprocess_shell(cmd, stdout=f_out, stderr=f_err, cwd=cwd, env=env)
            full_info["pid"] = proc.pid
            logging_callback(proc.pid)
            with open(task_description_folder / "full_description.json", "w", encoding="utf-8") as f_task_desc:
                json.dump(
                    full_info,
                    f_task_desc,
                    indent=4,
                )
            await proc.wait()
            return proc
