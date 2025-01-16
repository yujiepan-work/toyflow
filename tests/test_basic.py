import json
import random
import shutil
from pathlib import Path

from toyflow.launcher import Job, Launcher

if __name__ == "__main__":
    dummy_dict = dict(a='simple', b=2.333, c=[
                      'A string with spaces', '  \n  a string with whitespaces on both sides   ', 4, 5.0, {"d": "value_d"}])
    dummy_dict_str = json.dumps(dummy_dict, indent=4)

    job = Job(cmd=["echo", dummy_dict_str], job_name="shlex test")
    print(job.cmd_list)
    print(job.cmd_str)

    job2 = Job(cmd=job.cmd_str)
    assert job2.cmd_str == job.cmd_str
    print(job2.cmd_list)

    def create_folder(folder):
        Path(folder).mkdir(exist_ok=True)

    tmp = Path('./.tmp')
    shutil.rmtree(tmp)

    for cmd_type in ['str', 'list']:
        jobs = []
        for _ in range(4):
            i = random.randint(0, 5)
            job = Job(
                cmd={
                    'list': ["sleep", f'{i}'],
                    'str': f"sleep {i}; echo {i}",
                }[cmd_type],
                cwd=".",
                log_dir=f"{tmp}/{cmd_type}_{i}",
                prepare_fn=create_folder,
                cuda_quantity=i,
                job_name='A simple Job with id=' * 5 + str(i),
                prepare_fn_args=(f"{tmp}/{i}",),
                apply_shlex_parsing_for_cmd=True if cmd_type == 'list' else False,
                extra_info={
                    'extra_info': 'extra_info_{}'.format(i),
                }
            )
            jobs.append(job)
        print(jobs[-1].cmd_str)

        Launcher(
            list(range(8)), jobs,
            add_timestamp_to_log_dir=True,
            disable_env_info=False,
        ).start()

    # for p in os.walk(tmp):
    #     print(p)
