import os
import subprocess
from pathlib import Path

from setuptools import find_packages, setup

repo_root = Path(__file__).parent


def find_version_info():
    try:
        git_output = subprocess.check_output(
            ["git", "rev-list", "--count", "HEAD"], cwd=repo_root)
        dev_version_id = git_output.strip().decode()
        return f"{int(dev_version_id):03d}"
    except Exception:
        return "000"


def get_commit_date():
    try:
        git_output = subprocess.check_output(
            ["git", "log", "-1", "--format=%cd", "--date=format:%y%m%d"],
            cwd=repo_root)
        return git_output.strip().decode()
    except Exception:
        return "000000"


if os.environ.get('TOYFLOW_ADD_DEV_VERSION', '1').lower() in ['1', 'true']:
    version = f'.dev{get_commit_date()}{find_version_info()}'
else:
    version = ''

setup(
    author="yujiepan",
    author_email="yujiepan@no-email.example.com",
    python_requires=">=3.8",
    description="Lightweight flow execution tool.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    license="MIT license",
    keywords="toyflow",
    name="toyflow",
    packages=find_packages(where="./src/", include=['*']),
    package_dir={"": "src"},
    url="https://github.com/yujiepan-work/toyflow",
    version=f"0.2.1" + version,
    include_package_data=True,
    package_data={
        'toyflow': ['src/toyflow/callbacks/web_callback.html'],
    },
    install_requires=[
        "pandas",
        "flask",
    ],
)
