import datetime
import logging
import os
import socket
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import List

import pandas as pd
from flask import Flask

from toyflow.callbacks.base import Callback
from toyflow.job import Job

logging.basicConfig(level=logging.INFO)

with open(Path(__file__).parent / 'web_callback.html', 'r', encoding='utf-8') as f:
    INDEX_HTML = f.read()


class WebCallback(Callback):
    def __init__(self, config) -> None:
        super().__init__(config)
        self.jobs = []

    def on_launcher_start(self, jobs: List[Job]):
        self.jobs = [*jobs]
        self.port = self._get_free_port(start_port=30088)
        self.app = Flask('Web')
        self.app.logger.setLevel(logging.ERROR)
        logging.getLogger('werkzeug').setLevel(logging.ERROR)
        self._setup_routes()
        self.server_thread = threading.Thread(target=self._start_server)
        self.server_thread.daemon = True
        self.server_thread.start()
        logging.warning(f'Web server at: http://localhost:{self.port}')

    def on_launcher_end(self, jobs: List[Job]):
        if self.server_thread.is_alive():
            self.server_thread.join(timeout=1)

    def on_job_start(self, job: Job):
        job._start_time = datetime.now().isoformat()

    def on_job_end(self, job: Job):
        job._end_time = datetime.now().isoformat()

    def _setup_routes(self):
        self.app.add_url_rule(
            '/jobs', 'get_job_info', self.get_job_info, methods=['GET'])
        self.app.add_url_rule(
            '/', 'index', self.get_index_page, methods=['GET'])

    def _get_free_port(self, start_port=30088):
        port = start_port
        while True:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                if s.connect_ex(('localhost', port)) != 0:
                    return port
                port += 1

    def _start_server(self):
        self.app.run(port=self.port)

    def get_job_info(self):
        data = []
        for job in self.jobs:
            duration = ''
            if job._start_time:
                start = datetime.fromisoformat(job._start_time)
                end = datetime.fromisoformat(
                    job._end_time) if job._end_time else datetime.now()
                duration = (end - start).seconds
                duration = str(timedelta(seconds=duration))

            item = {
                'ID': job._job_id,
                'CUDA': str(job._resource.get_cuda_ids()).replace(' ', ''),
                'Name': job.job_name,
                'Status': job.status.name,
                'PID': job._pid,
                'Duration': duration,
            }
            if job._start_time:
                item['Start'] = datetime.fromisoformat(
                    job._start_time).strftime("%m%d-%H:%M:%S")
            if job._end_time:
                item['End'] = datetime.fromisoformat(
                    job._end_time).strftime("%m%d-%H:%M:%S")
            data.append(item)
        df = pd.DataFrame(data)
        return {
            'html': df.to_html(classes='table table-striped', index=False, escape=True, table_id="job-table"),
        }

    def get_index_page(self):
        return INDEX_HTML.replace(
            '[TITLE]',
            f"Tasks - {os.uname().nodename}"
        )


if __name__ == "__main__":
    jobs = [
        Job(cmd='sleep 100', job_name=f"Job {i}", _pid=i) for i in range(5)]
    callback = WebCallback()
    callback.on_launcher_start(jobs)
    input()
