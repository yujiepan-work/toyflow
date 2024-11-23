import asyncio
import logging
from typing import List, Optional

from toyflow.callbacks import (Callback, CompositeCallback, LoggingCallback,
                               RichCallback, WebCallback)
from toyflow.job import Job, JobStatus
from toyflow.resource import Resource, ResourceItem, ResourcePool, ResourceType
from toyflow.scheduler import JobScheduler

logging.basicConfig(level=logging.INFO)


class Launcher:
    def __init__(
        self,
        cuda_list: list[int],
        jobs: List[Job],
        callbacks: Optional[List[Callback]] = None,
        **kwargs,
    ):
        resource_items = [ResourceItem(ResourceType.CPU, 0, float('inf'))]
        for cuda_id in cuda_list:
            resource_items.append(ResourceItem(
                ResourceType.CUDA, int(cuda_id), 1.0
            ))
        self.resource_pool = ResourcePool(resource_items)

        all_callbacks = [
            LoggingCallback.from_config(**kwargs),
            RichCallback.from_config(**kwargs),
            WebCallback.from_config(**kwargs),
        ]
        if callbacks:
            all_callbacks.extend(callbacks)

        self.callback: CompositeCallback = CompositeCallback(
            callbacks=all_callbacks)
        self.job_scheduler = JobScheduler(jobs)
        self.resource_release_event = asyncio.Event()

    async def _run_job_and_then_release_resource(self, job: Job, resources: Resource):
        job._resource = resources
        job.env['CUDA_VISIBLE_DEVICES'] = ','.join(
            map(str, resources.get_cuda_ids())
        )
        retries_left = 1

        self.callback.on_job_start(job)

        with self.callback.during_job_context(job):
            while retries_left > 0 and job.status != JobStatus.FINISHED:
                process = await asyncio.create_subprocess_shell(
                    job.cmd_str,
                    env=job.env,
                    stdout=job._stdout, stderr=job._stderr,
                    cwd=job.cwd,
                    shell=True,
                )
                job._pid = process.pid
                self.callback.on_process_start(job, process)
                await process.wait()

                if process.returncode == 0:
                    self.job_scheduler.update_job(job, JobStatus.FINISHED)
                else:
                    retries_left -= 1
                self.callback.on_process_end(job, process)

            if job.status != JobStatus.FINISHED:
                job.status = JobStatus.FAILED
                logging.error(
                    f"Task {job.job_name} failed after retries.")

        self.callback.on_job_end(job)
        await self.resource_pool.release(resources)
        self.resource_release_event.set()

    async def _wait_for_next_proposal(self, timeout: float = 5):
        try:
            await asyncio.wait_for(self.resource_release_event.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            pass
        self.resource_release_event.clear()

    async def _start(self):
        running_tasks = []
        self.callback.on_launcher_start(self.job_scheduler.jobs)
        while self.job_scheduler.has_pending_jobs():
            available_resource = await self.resource_pool.allocate_all()
            sub_resource = None
            job = self.job_scheduler.get_next_job(available_resource)
            if job is not None:
                sub_resource = available_resource.split(
                    {ResourceType.CUDA: [1.0] * job.cuda_quantity})
            if job is None or sub_resource is None:
                await self.resource_pool.release(available_resource)
                await self._wait_for_next_proposal(timeout=5)
                continue

            available_resource.minus_(sub_resource)
            await self.resource_pool.release(available_resource)
            self.job_scheduler.update_job(job, JobStatus.RUNNING)

            running_task = asyncio.create_task(
                self._run_job_and_then_release_resource(job, sub_resource)
            )
            running_tasks.append(running_task)

        await asyncio.gather(*running_tasks)
        self.callback.on_launcher_end(self.job_scheduler.jobs)

    def start(self):
        asyncio.run(self._start())
        logging.info('Finished')
