import asyncio
import contextlib
import datetime
import logging
import platform
import sys
from pathlib import Path
from typing import TypeVar

from toyflow.callbacks.base import Callback
from toyflow.job import Job
from toyflow.utils.json_util import dump_json, load_json
from toyflow.utils.python_env import (get_conda_env_info, get_git_info,
                                      get_pip_editable_packages_with_git_info)

logging.basicConfig(level=logging.INFO)

Task = TypeVar('Task')

LOG_FOLDER_NAME = '.job-log'
ENV_FILENAME = 'job_env.json'
JOB_STATUS_FILENAME = 'job_info.json'


class LoggingCallback(Callback):

    def __init__(self) -> None:
        super().__init__()
        self._storage = {}

    def on_job_start(self, job: Job):
        log_dir = Path(job.log_dir, LOG_FOLDER_NAME)
        log_dir.mkdir(parents=True, exist_ok=True)

        dump_json(
            self.get_python_env_info(job),
            Path(log_dir, ENV_FILENAME)
        )

    @contextlib.contextmanager
    def during_job_context(self, job):
        with self.redirect_output(job):
            yield

    def on_process_start(self, job: Job, process: asyncio.subprocess.Process):
        path = Path(job.log_dir, LOG_FOLDER_NAME, JOB_STATUS_FILENAME)
        info = self.get_job_argv(job)
        info['host'] = platform.uname()._asdict(),
        info['launch_time'] = datetime.datetime.now().isoformat()
        info['end_time'] = None
        info['elapsed_time'] = None
        info['pid'] = process.pid
        info['job_status'] = job.status.name
        info['returncode'] = None
        dump_json(info, path)
        if 'running_info' not in self._storage:
            self._storage['running_info'] = {}
        self._storage['running_info'][job] = info

    def on_process_end(self, job: Job, process: asyncio.subprocess.Process):
        path = Path(job.log_dir, LOG_FOLDER_NAME, JOB_STATUS_FILENAME)
        flag = False
        if 'running_info' in self._storage and job in self._storage['running_info']:
            info = self._storage['running_info'][job]
            flag = True
        elif path.exists():
            info = load_json(path)
        else:
            info = {}

        info['end_time'] = datetime.datetime.now().isoformat()
        try:
            info['elapsed_time'] = str(
                datetime.datetime.now() -
                datetime.datetime.fromisoformat(info['launch_time'])
            )
        except:
            pass
        info['returncode'] = process.returncode
        info['job_status'] = job.status.name
        dump_json(info, path)
        if flag:
            del self._storage['running_info'][job]

    @contextlib.contextmanager
    def redirect_output(self, job: Job):
        stdout_path = Path(job.log_dir, LOG_FOLDER_NAME, f"stdout.log")
        stderr_path = Path(job.log_dir, LOG_FOLDER_NAME, f"stderr.log")
        with open(stdout_path, 'w', encoding='utf-8') as stdout, \
                open(stderr_path, 'w', encoding='utf-8') as stderr:
            job._stdout = stdout
            job._stderr = stderr
            yield
            job._stdout = sys.stdout
            job._stderr = sys.stderr

    def get_job_argv(self, job: Job):
        info = {}
        info['cwd'] = Path(job.cwd).as_posix()
        info['cmd'] = job.cmd_list
        info['log_dir'] = Path(job.log_dir).as_posix()
        info['job_name'] = job.job_name
        info['env'] = job._env_str
        info['resource'] = job._resource.as_dict()
        return info

    def get_python_env_info(self, job):
        info = {}
        info['conda'] = get_conda_env_info()
        info['pip_editable'] = get_pip_editable_packages_with_git_info()
        info['env'] = dict(sorted(job.env.items()))
        info['cwd_git_diff'] = get_git_info(job.cwd)
        info['cwd_git_diff']['cwd'] = Path(job.cwd).as_posix()
        return info

    def on_job_end(self, job: Job):
        return super().on_job_end(job)
