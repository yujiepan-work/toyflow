import asyncio
import heapq
import json
import logging
import os
import subprocess
from contextlib import asynccontextmanager
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union
from unittest.mock import Mock
import time
import platform

from toyflow.dag import Node


class Task(Node):
    def __init__(
        self,
        cmd: Union[str, List[Any]],
        cwd: Union[str, Path, os.PathLike],
        io_folder: Union[str, Path],
        identifier: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
        cuda_quantity: int = 1,
        cpu_quantity: float = 0.01,
        prepare_fn: Optional[Callable] = None,
        prepare_fn_args: Optional[tuple] = None,
    ) -> None:
        super().__init__(str(identifier))
        self.cmd = cmd
        self.cwd = cwd
        self.io_folder = Path(io_folder).resolve()
        self.env = env or os.environ.copy()
        self.cuda_quantity = cuda_quantity
        self.cpu_quantity = cpu_quantity
        self.prepare_fn = prepare_fn or Mock()
        self.prepare_fn_args = prepare_fn_args or tuple()
        self.identifier = str(identifier) or self.cmd_str()
        self.task_id = -1

    def cmd_str(self):
        cmd_l = self.cmd.split() if isinstance(self.cmd, str) else self.cmd
        return " ".join(map(str, cmd_l))

    def cmd_list(self):
        return self.cmd_str().split()

    def cmd_bash(self):
        return " \\\n    ".join(x.strip() for x in self.cmd_list())
