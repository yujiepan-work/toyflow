import random
import shutil
from pathlib import Path

from toyflow.launcher import Job, Launcher

if __name__ == "__main__":
    def create_folder(folder):
        Path(folder).mkdir(exist_ok=True)

    jobs = []
    tmp = Path('./.tmp')
    shutil.rmtree(tmp)

    for i in range(4):
        i = random.randint(0, 5)
        job = Job(
            cmd=["sleep", str(i), ";", "echo", str(i)],
            cwd=".",
            log_dir=f"{tmp}/{i}",
            prepare_fn=create_folder,
            cuda_quantity=i,
            job_name='A simple Job with id=' * 5 + str(i),
            prepare_fn_args=(f"{tmp}/{i}",),
            extra_info={
                'extra_info': 'extra_info_{}'.format(i),
            }
        )
        jobs.append(job)

    Launcher(list(range(8)), jobs, add_timestamp_to_log_dir=True).start()
    # for p in os.walk(tmp):
    #     print(p)
