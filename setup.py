import subprocess
from pathlib import Path

from setuptools import find_packages, setup

repo_root = Path(__file__).parent


def find_version_info():
    try:
        git_output = subprocess.check_output(["git", "rev-list", "--count", "HEAD"], cwd=repo_root)
        dev_version_id = git_output.strip().decode()
        return f".dev{dev_version_id}"
    except Exception:
        return ""


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
    packages=find_packages(include=["toyflow", "toyflow.*"]),
    url="https://github.com/yujiepan-work/toyflow",
    version=f"0.2.0{find_version_info()}",
)
