import asyncio
import contextlib
import logging
from dataclasses import dataclass
from typing import List, TypeVar

from toyflow.job import Job

logging.basicConfig(level=logging.INFO)

Task = TypeVar('Task')

LOG_FOLDER_NAME = '.job-info'


@dataclass
class CallbackConfig:
    pass


class Callback:
    config_cls: dataclass = CallbackConfig

    @classmethod
    def from_config(cls, **kwargs):
        valid_keys = {
            field.name for field in cls.config_cls.__dataclass_fields__.values()}
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_keys}
        config = cls.config_cls(**filtered_kwargs)
        logging.info(f'Registered {cls.__name__}(config={config})')
        obj = cls(config)
        return obj

    def __init__(self, config: CallbackConfig):
        self.config = config

    def on_launcher_start(self, jobs: List[Job]):
        pass

    def on_launcher_end(self, jobs: List[Job]):
        pass

    def on_job_start(self, job: Job):
        pass

    def on_job_end(self, job: Job):
        pass

    @contextlib.contextmanager
    def during_job_context(self, job: Job):
        yield

    def on_process_start(self, job: Job, process: asyncio.subprocess.Process):
        pass

    def on_process_end(self, job: Job, process: asyncio.subprocess.Process):
        pass


class CompositeCallback(Callback):
    def __init__(self, config=Callback.config_cls(), callbacks: List[Callback] = tuple()):
        super().__init__(config)
        self.callbacks = callbacks

    def on_launcher_start(self, jobs: List[Job]):
        for callback in self.callbacks:
            callback.on_launcher_start(jobs)

    def on_launcher_end(self, jobs: List[Job]):
        for callback in self.callbacks:
            callback.on_launcher_end(jobs)

    def on_job_start(self, job: Job):
        for callback in self.callbacks:
            callback.on_job_start(job)

    def on_job_end(self, job: Job):
        for callback in self.callbacks:
            callback.on_job_end(job)

    @contextlib.contextmanager
    def during_job_context(self, job: Job):
        with contextlib.ExitStack() as stack:
            for callback in self.callbacks:
                stack.enter_context(callback.during_job_context(job))
            yield

    def on_process_start(self, job: Job, process: asyncio.subprocess.Process):
        for callback in self.callbacks:
            callback.on_process_start(job, process)

    def on_process_end(self, job: Job, process: asyncio.subprocess.Process):
        for callback in self.callbacks:
            callback.on_process_end(job, process)
