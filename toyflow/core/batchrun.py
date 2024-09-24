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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from unittest.mock import Mock

import rich
from rich.console import Console
from rich.progress import (BarColumn, Progress, TaskProgressColumn, TextColumn,
                           TimeElapsedColumn, TimeRemainingColumn)
from rich.style import StyleType

from toyflow.core.task_node import Task
from toyflow.dag import TopoSorter
from toyflow.resources import CPUResource, CUDAResource, Resource
from toyflow.utils import logger

console = Console()


class MockLogger():
    def info(self, *args, **kwargs):
        return self.rich_print(*args, **kwargs)

    def warning(self, *args, **kwargs):
        return self.rich_print(*args, **kwargs)

    def error(self, *args, **kwargs):
        return self.rich_print(*args, **kwargs)

    def critical(self, *args, **kwargs):
        return self.rich_print(*args, **kwargs)

    def rich_print(self, *args, **kwargs):
        console.print(args[0] % (args[1:]), **kwargs)


logger = MockLogger()

CUDA_VISIBLE_DEVICES = "CUDA_VISIBLE_DEVICES"


def pre_launch_worker(io_folder):
    Path(io_folder).mkdir()


def task_sort_key(task: Task):
    return (task.cuda_quantity, task.cpu_quantity)


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
        self.cuda_resource = CUDAResource(cuda_list)
        self._lock = asyncio.Lock()
        self.check_interval = 1  # check every 1 sec
        self._cuda_str_max_length = 3  # len(",".join(map(str, cuda_list)))

    def run(self, tasks: List[Task], add_timestamp_to_log=False):
        self.add_timestamp_to_log = add_timestamp_to_log
        self.progress_tasks = {}
        self.progress = Progress(
            TextColumn("{task.fields[cuda_list]}"),
            TextColumn("{task.description}"),
            TextColumn('{task.fields[status]}'),
            BarColumn(pulse_style='bar.back'),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TextColumn("{task.start_time}"),
            TextColumn("{task.stop_time}"),
            refresh_per_second=1.0/3.5,
            redirect_stderr=False,
            redirect_stdout=False,
            expand=False,
            console=console,
        )
        with self.progress:
            for i, task in enumerate(tasks):
                task.task_id = i + 1
                self.progress_tasks[task] = self.progress.add_task(str(task.identifier),
                                                                   start=False, total=1, cuda_list=[], status='PENDING')

            runner = asyncio.run(self._run_v2(tasks))
            # runner = asyncio.run(self._run(tasks))
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

            def logging_callback(pid): return logger.info(
                "Running Task[%d/%s] PID=%d CUDA=%s: %s",
                task_id,
                total,
                pid,
                cuda.ljust(self._cuda_str_max_length),
                task.identifier,
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
                cuda.ljust(self._cuda_str_max_length),
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
        timestamp = ("_" + str(int(time.time()))
                     ) if self.add_timestamp_to_log else ""
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

    def _select_next_task(self, candidates: List[Task], cuda_resource: asyncio.Queue):
        if not candidates:
            return None
        task = sorted(candidates, key=task_sort_key)[0]
        if (cuda_resource.qsize()) >= task.cuda_quantity:
            return task
        else:
            return None

    async def return_back_resource(self, cuda_list):
        async with self._lock:
            for item in cuda_list:
                await self.cuda_resource.resources.put(item)

    async def try_launch_next_task(self):
        async with self._lock:
            candidates = [t for t in self.runner.get_next_node_candidates(
            ) if t not in self.launched_tasks]
            task = self._select_next_task(
                candidates, self.cuda_resource.resources)
            if task is None:
                return
            cuda_list = await self.cuda_resource.allocate_no_recycle(task.cuda_quantity, timeout=None)
            if cuda_list is None:
                return
            self.launched_tasks.append(task)

        async_task = asyncio.create_task(
            self._launch_task_and_then_return_back_resource(
                task,
                cuda_list,
            )
        )
        # async_task.add_done_callback(self.launched_tasks.remove)

    async def _run_v2(self, tasks):
        self.runner = TopoSorter.from_nodes(tasks)
        self.launched_tasks = list()
        await self.try_launch_next_task()  # trigger the first task
        while len(self.runner.get_remaining_nodes()) > 0:
            await self.try_launch_next_task()
            await asyncio.sleep(self.check_interval)
        return self.runner

    async def _launch_task_and_then_return_back_resource(
        self,
        task: Task,
        cuda_list,
    ):
        self.progress.start_task(self.progress_tasks[task])
        self.progress.update(
            self.progress_tasks[task], cuda_list=cuda_list, status='RUNNING')
        status = await self._launch_task(task, task.task_id, None, cuda_list=cuda_list, total=self.runner.graph.num_nodes)
        async with self._lock:
            self.runner.set_next_node(task)  # set this task as finished
        await self.return_back_resource(cuda_list)
        self.progress.update(
            self.progress_tasks[task], completed=1, status=status)
        self.progress.stop_task(self.progress_tasks[task])
        # await self.try_launch_next_task()
