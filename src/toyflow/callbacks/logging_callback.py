import asyncio
import contextlib
import datetime
import logging
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

from toyflow.callbacks.base import Callback
from toyflow.job import Job
from toyflow.utils.json_util import dump_json, load_json
from toyflow.utils.python_env import (get_conda_env_info,
                                      get_environment_variables, get_git_info,
                                      get_pip_editable_packages_with_git_info,
                                      get_simple_conda_env_info)

logging.basicConfig(level=logging.INFO)

Task = TypeVar('Task')


@dataclass
class LoggingCallbackConfig:
    log_folder_name: str = '.job-log'
    env_filename: str = 'job_env.json'
    job_info_filename: str = 'job_info.json'
    add_timestamp_to_log_dir: bool = False
    dependency_keywords: list[str] = (
        'torch', 'transformers', 'openvino', 'accelerate', 'cuda=',
        'diffusers', 'optimum', 'nncf', 'vllm',
        'sglang', 'sgl-kernel', 'triton', 'flashinfer',
    )
    remove_sensitive_env_keys: bool = True
    remove_sensitive_env_keys_extra_list: list[str] = ()
    force_only_show_selected_env_keys: bool = True
    force_only_show_env_keys_extra_list: list[str] = ()
    disable_env_info: bool = False


class LoggingCallback(Callback):
    config_cls = LoggingCallbackConfig

    def __init__(
        self, config: LoggingCallbackConfig,
    ) -> None:
        super().__init__(config)
        self.config: LoggingCallbackConfig
        self._storage = {}

    def on_job_start(self, job: Job):
        log_dir = Path(job.log_dir, self.config.log_folder_name)
        log_dir.mkdir(parents=True, exist_ok=True)

        dump_json(
            self.get_python_env_info(job),
            Path(log_dir, self.config.env_filename)
        )

    @contextlib.contextmanager
    def during_job_context(self, job):
        with self.redirect_output(job):
            yield

    def on_process_start(self, job: Job, process: asyncio.subprocess.Process):
        path = Path(job.log_dir, self.config.log_folder_name,
                    self.config.job_info_filename)
        info = self.get_job_argv(job)
        info['extra_info'] = job.extra_info

        if not self.config.disable_env_info:
            info['conda'] = get_simple_conda_env_info(
                keywords=self.config.dependency_keywords
            )
            info['cwd_git_diff'] = get_git_info(job.cwd, return_diff=False)
            info['cwd_git_diff']['cwd'] = Path(job.cwd).as_posix()

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
        path = Path(job.log_dir, self.config.log_folder_name,
                    self.config.job_info_filename)
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
        if self.config.add_timestamp_to_log_dir:
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            stdout_path = Path(job.log_dir, f"stdout_{timestamp}.log")
            stderr_path = Path(job.log_dir, f"stderr_{timestamp}.log")
        else:
            stdout_path = Path(job.log_dir, f"stdout.log")
            stderr_path = Path(job.log_dir, f"stderr.log")
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
        if self.config.disable_env_info:
            return info
        info['conda'] = get_conda_env_info()
        info['pip_editable'] = get_pip_editable_packages_with_git_info()
        info['env'] = get_environment_variables(
            job.env, remove_sensitive=self.config.remove_sensitive_env_keys,
            remove_sensitive_extra_list=self.config.remove_sensitive_env_keys_extra_list,
            force_only_show_selected_env_keys=self.config.force_only_show_selected_env_keys,
            force_only_show_env_keys_extra_list=self.config.force_only_show_env_keys_extra_list,
        )
        info['cwd_git_diff'] = get_git_info(job.cwd)
        info['cwd_git_diff']['cwd'] = Path(job.cwd).as_posix()
        return info

    def on_job_end(self, job: Job):
        return super().on_job_end(job)
