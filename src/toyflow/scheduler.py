import logging
from typing import List, Optional

from toyflow.job import Job, JobStatus
from toyflow.resource import Resource, ResourceType

logging.basicConfig(level=logging.INFO)


class JobScheduler:
    def __init__(self, jobs: List[Job]) -> None:
        self.jobs = jobs

    def has_pending_jobs(self):
        for job in self.jobs:
            if job.status == JobStatus.PENDING:
                return True
        return False

    def get_next_job(self, available_resource: Resource) -> Optional[Job]:
        remaining_jobs = sorted(
            [job for job in self.jobs if job.status == JobStatus.PENDING],
            key=lambda job: (job.cuda_quantity, -job._job_id, job.cpu_quantity)
        )
        if not remaining_jobs:
            return None
        for job in remaining_jobs[::-1]:
            if len(list(k for k, v in available_resource[ResourceType.CUDA].items() if v > 0)) < job.cuda_quantity:
                continue
            return job
        return None

    def update_job(self, job: Job, new_status: JobStatus):
        job.status = new_status
