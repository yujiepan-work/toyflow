import asyncio
import contextlib
import logging
from typing import List, TypeVar

from toyflow.job import Job

logging.basicConfig(level=logging.INFO)

Task = TypeVar('Task')

LOG_FOLDER_NAME = '.job-info'


class Callback:
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
    def __init__(self, callbacks: List[Callback]):
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
