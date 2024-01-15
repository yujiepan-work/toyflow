import asyncio
import functools
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
from enum import Enum, IntEnum
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
from toyflow.utils.constants import (CUDA_VISIBLE_DEVICES, SESSION_ID,
                                     TASK_DETAIL_INFO_FILENAME,
                                     TASK_INFORMATION_FOLDER_NAME,
                                     TASK_SIMPLE_INFO_FILENAME)

console = Console()


def dump_json(obj, file_path: Union[str, os.PathLike, Path]):
    file_path = Path(file_path)
    file_path.parent.mkdir(exist_ok=True, parents=True)
    assert file_path.name.lower().endswith('.json')
    if file_path.exists():
        file_path = file_path.parent / f'{file_path.stem}.{SESSION_ID}.json'
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2)
    return file_path


class TaskStatus(IntEnum):
    UNKNOWN = -1
    PENDING = 0
    LAUNCHING = 1
    RUNNING = 2
    FINISHING = 3
    SUCCESS = 4
    FAILED = 5


@dataclass
class ResourceResponse:
    status_ok: bool
    resource: Any = None


class ResourceManager:
    _lock = asyncio.Lock()

    def __init__(self, cuda_list) -> None:
        self.resources = []
        self.cuda_resource = cuda_list

    async def return_back_resource(self, cuda_list):
        async with self._lock:
            for item in cuda_list:
                self.cuda_resource.append(item)

    async def get_resource(self, cuda_quantity: int):
        async with self._lock:
            if len(self.cuda_resource) >= cuda_quantity:
                resource = [self.cuda_resource.pop() for _ in range(cuda_quantity)]
                return ResourceResponse(status_ok=True, resource=resource)
            else:
                return ResourceResponse(status_ok=False)


class TaskCallback:
    def on_start(self, task: Task):
        pass

    def on_end(self, task: Task):
        pass

    def on_success(self, task: Task):
        pass

    def on_failed(self, task: Task):
        pass


class TaskCallbacksHandler:
    def __init__(self) -> None:
        self.callbacks: list[TaskCallback] = []

    def on_start(self, task):
        for cb in self.callbacks:
            cb.on_start(task)

    def on_end(self, task):
        for cb in self.callbacks:
            cb.on_end(task)

    def on_success(self, task):
        for cb in self.callbacks:
            cb.on_success(task)

    def on_failed(self, task):
        for cb in self.callbacks:
            cb.on_failed(task)


class TaskDesctriptionCallback(TaskCallback):
    def on_start(self, task: Task):
        io_folder = Path(task.io_folder)
        env = task.env
        io_folder.mkdir(exist_ok=True, parents=True)
        task_description_folder = io_folder / TASK_INFORMATION_FOLDER_NAME
        task_description_folder.mkdir(exist_ok=True, parents=True)
        full_info = {
            "cmd_str": task.cmd_str(),
            "cwd": Path(task.cwd).absolute().as_posix(),
            "cmd_list": task.cmd_list(),
            "env": dict(sorted(env.items())),
            "host": platform.uname()._asdict(),
            "launch_time": time.localtime(),
        }
        actual_detail_path = dump_json(full_info, task_description_folder / TASK_DETAIL_INFO_FILENAME)
        task.actual_detail_path = actual_detail_path
        task.task_description_folder = task_description_folder

        cwd = Path(task.cwd).absolute()
        simple_info = {
            "cwd": cwd.relative_to(Path.home()).as_posix()
            if cwd.is_relative_to(Path.home())
            else cwd.as_posix(),
            "cmd_list": task.cmd_list(),
        }
        dump_json(simple_info, io_folder / TASK_SIMPLE_INFO_FILENAME)
        task.full_info = full_info

    def on_end(self, task: Task):
        task.full_info["end_time"] = time.localtime()
        dump_json(task.full_info, task.actual_detail_path)


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


def pre_launch_worker(io_folder):
    Path(io_folder).mkdir()


def task_sort_key(task: Task):
    return (task.cuda_quantity, task.cpu_quantity)


class Launcher:
    def __init__(self, cuda_list: List[int]) -> None:
        print("GOOD LUCK")
        self.cuda_list = cuda_list
        self.add_timestamp_to_log = False
        self.resource_manager = ResourceManager(cuda_list)
        self.check_interval = 1  # check every 1 sec
        self._cuda_str_max_length = 3  # len(",".join(map(str, cuda_list)))
        self._lock = asyncio.Lock()

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
            refresh_per_second=1.0 / 3.5,
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

    async def _launch_task(self, task: Task, task_id: int, resource_manager: CUDAResource, **kwargs) -> str:
        total = str(kwargs.get("total", "?"))
        cuda_list = kwargs.get("cuda_list", tuple())
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
            task.env = env

            cb = TaskDesctriptionCallback()
            cb.on_start(task)

            start_time = time.time()
            proc = await self._run_single_process(
                task.cmd_str(), task.io_folder, task.cwd, env, task.task_description_folder, task.full_info, logging_callback
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

            cb.on_end(task)

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

    async def try_launch_next_task(self):
        candidates = [t for t in self.runner.get_next_node_candidates() if t not in self.launched_tasks]
        if not candidates:
            return

        task: Task = sorted(candidates, key=task_sort_key)[0]
        response = await self.resource_manager.get_resource(task.cuda_quantity)
        if not response.status_ok:
            return

        self.launched_tasks.append(task)
        async_task = asyncio.create_task(
            self._launch_task_and_then_return_back_resource(
                task,
                response.resource,
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

        await self.resource_manager.return_back_resource(cuda_list)
        self.progress.update(
            self.progress_tasks[task], completed=1, status=status)
        self.progress.stop_task(self.progress_tasks[task])
        # await self.try_launch_next_task()
