from pathlib import Path

from toyflow import Launcher, Task

if __name__ == "__main__":
    launcher = Launcher([1, 2, 3, 4, 5])
    tasks: list[Task] = []

    def create_folder(folder):
        Path(folder).mkdir(exist_ok=True)

    for i in range(5):
        task = Task(
            cmd=["sleep", str(i), "&&", "echo", str(i)],
            cwd=".",
            io_folder=f"tmp/{i}",
            prepare_fn=create_folder,
            cuda_quantity=i,
            identifier='A simple task with id=' + str(i),
            prepare_fn_args=(f"/tmp/{i}",),
        )
        tasks.append(task)

    tasks[4].add_downstreams(tasks[0])
    tasks[3].add_downstreams([tasks[2], tasks[1]])
    launcher.run(tasks, add_timestamp_to_log=False)
    (3, 1, 2, 4, 0)
