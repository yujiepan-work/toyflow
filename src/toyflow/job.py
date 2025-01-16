import logging
import os
import shlex
import sys
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union
from unittest.mock import Mock

from toyflow.resource import Resource

logging.basicConfig(level=logging.INFO)


class JobStatus(IntEnum):
    PENDING = 0
    LAUNCHING = 1
    RUNNING = 2
    FAILED = 3
    FINISHED = 4


@dataclass
class Job:
    cmd: Union[str, List[Any]]
    cwd: Union[str, Path, os.PathLike] = '.'
    log_dir: Union[str, Path, os.PathLike] = '.'
    job_name: Optional[str] = None
    env: Optional[Dict[str, str]] = None
    cuda_quantity: int = 1
    cpu_quantity: float = 0.01
    prepare_fn: Optional[Callable] = None
    prepare_fn_args: Optional[tuple] = None
    status: JobStatus = JobStatus.PENDING
    extra_info: Dict[str, Any] = field(default_factory=dict)
    apply_shlex_parsing_for_cmd: bool = True
    _stdout = sys.stdout
    _stderr = sys.stderr
    _resource: Resource = field(default_factory=Resource)
    _job_id: int = -1
    _pid: int = -1
    _start_time: str = ''
    _end_time: str = ''

    def __post_init__(self):
        self.cwd = Path(self.cwd).resolve()
        self.log_dir = Path(self.log_dir).resolve()
        self._env_str = '<from parent>' if self.env is None else "<customized>"
        self.env = self.env or os.environ.copy()
        self.job_name = str(self.job_name) if self.job_name else None
        self.prepare_fn = self.prepare_fn or Mock()

        if not self.apply_shlex_parsing_for_cmd:
            assert isinstance(self.cmd, str), \
                "cmd must be a string if `apply_shlex_parsing_for_cmd` is False."
            logging.warning((
                "You have set `apply_shlex_parsing_for_cmd` to False. "
                "In this case, `cmd` must be a string, and will be used directly "
                "as the command to run in shell. Use with care."
            ))

        # Call these two methods to check if the cmd is valid.
        self.cmd_list, self.cmd_str

    @property
    def cmd_str(self) -> str:
        """Returns the command as a single string."""
        if not self.apply_shlex_parsing_for_cmd:
            return str(self.cmd)

        result = shlex.join(self.cmd_list)
        if any((
            "&'" in result,
            ";'" in result,
        )):
            logging.warning((
                "You might be using shell features like `&` or `;` in the command. "
                "They are regarded as normal string in the command to run, which "
                "might lead to unexpected behavior. "
                "If you are sure about what you are doing, "
                "please set `apply_shlex_parsing_for_cmd` to False, "
                "and pass a string to `cmd` directly."
            ))
        return result

    @property
    def cmd_list(self) -> List[str]:
        """Returns the command as a list of strings."""
        if isinstance(self.cmd, str):
            return shlex.split(self.cmd)

        assert isinstance(self.cmd, list), "cmd must be a string or a list"
        result: List[str] = []
        for item in self.cmd:
            if isinstance(item, str):
                result.append(item)
            else:
                logging.warning(
                    f"Non-string item {item} in cmd list is converted by calling `str(item)`.")
                result.append(str(item))

        result_str = shlex.join(result)
        return shlex.split(result_str)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"cmd={self.cmd_list}, "
            f"cwd='{self.cwd}', "
            f"env={self._env_str}, "
            f"job_name='{self.job_name}', "
            f"status={self.status.name})"
        )

    def __hash__(self) -> int:
        return id(self)
