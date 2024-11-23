import datetime
import logging
from asyncio.subprocess import Process
from datetime import datetime
from typing import List, TypeVar

import rich
import rich.layout
import rich.live
import rich.panel
from rich.console import Console
from rich.progress import (BarColumn, Progress, TaskProgressColumn, TextColumn,
                           TimeElapsedColumn)
from rich.text import Text

from toyflow.callbacks.base import Callback
from toyflow.job import Job

logging.basicConfig(level=logging.INFO)

Task = TypeVar('Task')

LOG_FOLDER_NAME = '.job-info'


console = Console()


class RichCallback(Callback):
    def __init__(self, config) -> None:
        super().__init__(config)
        self.console = console
        self._additional_info = Text('[Running...]')

    def on_launcher_start(self, jobs: List[Job]):
        self.job_vs_task_id = {}
        self.progress = Progress(
            TextColumn("{task.fields[job_id]}"),
            TextColumn("{task.fields[cuda_list]}"),
            TextColumn("{task.description}"),
            TextColumn('{task.fields[status]}'),
            TextColumn('{task.fields[pid]}'),
            BarColumn(pulse_style='bar.back'),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TextColumn("{task.fields[start_time_str]}"),
            TextColumn("{task.fields[stop_time_str]}"),
            refresh_per_second=0.33,
            redirect_stderr=False,
            redirect_stdout=False,
            expand=False,
            console=self.console,
        )
        self.live = rich.live.Live(
            self.layout(), console=self.console, refresh_per_second=0.33)
        self.live.start()

        for i, job in enumerate(jobs):
            job._job_id = i + 1
            self.job_vs_task_id[job] = self.progress.add_task(
                description=str(job.job_name),
                start=False, total=1,
                start_time_str='None',
                stop_time_str='None',
                job_id=i+1,
                pid=job._pid,
                cuda_list=[], status='PENDING',
            )

    def layout(self):
        from rich.layout import Layout
        result = Layout()
        result.split(
            Layout(self.progress),
            Layout(self._additional_info, size=1),
        )
        return result

    def on_launcher_end(self, jobs: List[Job]):
        self.live.stop()
        # display the progress table's string
        self.console.print('\n\n[Job Summary]')
        self.console.print(self.progress)

    def on_job_start(self, job: Job):
        self.progress.start_task(self.job_vs_task_id[job])
        self.progress.update(
            self.job_vs_task_id[job],
            cuda_list=str(job._resource.get_cuda_ids()).replace(' ', ''),
            status='RUNNING',
            start_time_str=datetime.now().strftime("%m%d-%H%M%S"),
            stop_time_str='None',
            pid=job._pid,
        )

    def on_process_start(self, job: Job, process: Process):
        self.progress.update(
            self.job_vs_task_id[job],
            pid=process.pid,
        )

    def update_log(self, text):
        self._additional_info = Text(text)
        self.live.update(self.layout())

    def on_job_end(self, job: Job):
        self.progress.update(
            self.job_vs_task_id[job], completed=1,
            status=job.status.name,
            stop_time_str=datetime.now().strftime("%m%d-%H%M%S"),
        )
        self.progress.stop_task(self.job_vs_task_id[job])
        stopped_tasks = len(
            [task for task in self.progress.tasks if task.completed])
        total_tasks = len(self.progress.tasks)
        self.update_log(
            f'[{stopped_tasks}/{total_tasks}] | Last finished ID: {job._job_id} -- {job}')
