# toyflow

Lightweight flow execution tool.

Work in progress.

```bash
pip install git+https://github.com/yujiepan-work/toyflow.git
```

#### Usage
```python
from toyflow.launcher import Job, Launcher

jobs = []
for i in range(5):
    job = Job(
        cmd=["sleep", str(i), ";", "echo", str(i)],
        cwd=".",
        log_dir=f"./tmp/{i}",
        cuda_quantity=1,
        job_name=f'Job #{i}',
    )
    jobs.append(job)

Launcher(cuda_list=[0, 1], jobs=jobs).start()
```


#### Note
If you find the interface has changed, you can install the older version: 
```bash
pip install git+https://github.com/yujiepan-work/toyflow.git@v0.1.0
```
