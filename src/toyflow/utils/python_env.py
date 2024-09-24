import json
import subprocess
from typing import Dict


def get_conda_env_info() -> Dict:
    result = subprocess.run(
        ['conda', 'env', 'export', '--json'], capture_output=True, text=True, check=True)
    conda_info = json.loads(result.stdout)
    return conda_info


def get_git_info(path):
    try:
        git_commit_id = subprocess.check_output(
            ['git', '-C', path, 'rev-parse', 'HEAD']).strip().decode('utf-8')
        git_reflog = subprocess.check_output(
            ['git', '-C', path, 'reflog']).strip().decode('utf-8')
        git_diff = subprocess.check_output(
            ['git', '-C', path, 'diff']).strip().decode('utf-8')
        if not git_diff:
            git_diff = "No uncommitted changes"
    except subprocess.CalledProcessError:
        git_commit_id = "Not a Git repository"
        git_reflog = "Not a Git repository"
        git_diff = "Not a Git repository"

    return {
        "git_commit_id": git_commit_id,
        "git_reflog": git_reflog,
        "uncommitted_changes_diff": git_diff
    }


def get_pip_editable_packages_with_git_info():
    result = subprocess.run(
        ['pip', 'list', '--editable', '--format=json'], capture_output=True, text=True)
    editable_packages = json.loads(result.stdout)
    packages_with_git_info = []
    for package in editable_packages:
        package_name = package['name']
        editable_project_location = package.get(
            'location') or package.get('editable_project_location')
        git_info = get_git_info(editable_project_location)
        packages_with_git_info.append({
            "name": package_name,
            "editable_project_location": editable_project_location,
            **git_info
        })
    return packages_with_git_info
