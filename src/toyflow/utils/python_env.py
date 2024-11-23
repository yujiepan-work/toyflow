import json
import os
import subprocess
from copy import deepcopy
from typing import Dict


def get_conda_env_info() -> Dict:
    result = subprocess.run(
        ['conda', 'env', 'export', '--json'], capture_output=True, text=True, check=True)
    conda_info = json.loads(result.stdout)
    return conda_info


def get_simple_conda_env_info(keywords: list[str]):
    obj: dict = deepcopy(get_conda_env_info())
    obj.pop('channels', None)
    conda_dependencies = [
        x for x in obj.get('dependencies', tuple())
        if isinstance(x, str) and
        any(k.lower() in x.lower() for k in keywords)
    ]
    pip_dict = obj.get('dependencies')[-1]
    assert isinstance(pip_dict, dict) and 'pip' in pip_dict
    pip_dependencies = [
        x for x in pip_dict['pip']
        if isinstance(x, str) and
        any(k.lower() in x.lower() for k in keywords)
    ]
    obj['dependencies'] = {
        'conda': conda_dependencies,
        'pip': pip_dependencies,
    }
    return obj


def get_git_info(path, return_diff=True):
    try:
        git_commit_id = subprocess.check_output(
            ['git', '-C', path, 'rev-parse', 'HEAD']).strip().decode('utf-8')
        git_reflog = subprocess.check_output(
            ['git', '-C', path, 'reflog', "-n", '5']).strip().decode('utf-8')
        git_diff = subprocess.check_output(
            ['git', '-C', path, 'diff']).strip().decode('utf-8')
        if not git_diff:
            git_diff = "No uncommitted changes"
    except subprocess.CalledProcessError:
        git_commit_id = "Not a Git repository"
        git_reflog = "Not a Git repository"
        git_diff = "Not a Git repository"
    output = {
        "git_commit_id": git_commit_id,
        "git_reflog": git_reflog.split('\n'),
    }
    if return_diff:
        output["uncommitted_changes_diff"] = git_diff.split('\n'),
    return output


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


def get_environment_variables(
    env: dict = None, remove_sensitive=True,
    remove_sensitive_extra_list: list[str] = (),
    force_only_show_selected_env_keys: bool = True,
    force_only_show_env_keys_extra_list: list[str] = (),
):
    env = deepcopy(env) or os.environ.copy()
    removed_keys = set()
    if remove_sensitive and (force_only_show_selected_env_keys is False):
        sensitive_keys = [
            "PROXY",
            "SOCKS",
            "PASSWORD",
            "API_KEY",
            "AUTO_TOKEN",
            "TOKEN",
            "SECRET",
            "CREDENTIAL",
            "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY",
            "AWS_SESSION_TOKEN",
            "GOOGLE_CLOUD_PROJECT",
            "GOOGLE_APPLICATION_CREDENTIALS",
            "AZURE_CLIENT_ID",
            "AZURE_CLIENT_SECRET",
            "AZURE_SUBSCRIPTION_ID",
            "AZURE_TENANT_ID",
            "GITHUB_TOKEN",
            "GH_TOKEN",
            "HEROKU_API_KEY",
            "DOCKER_USERNAME",
            "DOCKER_PASSWORD",
            "STRIPE_API_KEY",
            "STRIPE_SECRET_KEY",
            "TWILIO_ACCOUNT_SID",
            "TWILIO_AUTH_TOKEN",
            "SENDGRID_API_KEY",
            "SLACK_API_TOKEN",
            "SLACK_BOT_TOKEN",
            "FIREBASE_API_KEY",
            "FIREBASE_AUTH_DOMAIN",
            "FIREBASE_PROJECT_ID",
            "FIREBASE_STORAGE_BUCKET",
            "FIREBASE_MESSAGING_SENDER_ID",
            "FIREBASE_APP_ID",
            "PAYPAL_CLIENT_ID",
            "PAYPAL_CLIENT_SECRET",
            "MONGODB_URI",
            "MONGO_INITDB_ROOT_USERNAME",
            "MONGO_INITDB_ROOT_PASSWORD",
            "POSTGRES_USER",
            "POSTGRES_PASSWORD",
            "POSTGRES_DB",
            "REDIS_URL",
            "ALGOLIA_APP_ID",
            "ALGOLIA_API_KEY",
            "SENTRY_DSN",
            "MAILGUN_API_KEY",
            "MAILGUN_DOMAIN",
            "CONTENTFUL_SPACE_ID",
            "CONTENTFUL_ACCESS_TOKEN",
            "NETLIFY_AUTH_TOKEN",
            "CIRCLECI_API_TOKEN",
            "JENKINS_API_TOKEN",
            "WANDB_API_KEY",
            "HUGGINGFACE_HUB_TOKEN",
            "HF_HUB_TOKEN",
            "HF_TOKEN",
            "VSCODE",
            "XDG_",
            "LS_COLOR",
            "CONDA_",
            'GSETTINGS_',
            'AUTOJUMP_',
            'DBUS_SESSION_',
            'ASKPASS',
            "SSH_",
            "TERM_",
            "WANDB_",
            "ZSH",
            "LC_",
            "SSL_",
            *remove_sensitive_extra_list,
        ]
        for key in list(env.keys()):
            if any(secret.lower() in key.lower() for secret in sensitive_keys):
                env.pop(key, None)
                removed_keys.add(key)
    if force_only_show_selected_env_keys is True:
        only_show = [
            'CUDA_VISIBLE_DEVICES', 'PYTHONPATH', 'PATH', 'LD_LIBRARY_PATH',
            'DEVICES', 'CUDA_HOME', 'CONDA_PREFIX', 'HOSTNAME', 'TRITON_PTXAS_PATH',
            'HF_HOME', 'HF_ENDPOINT', 'HF_HUB_CACHE', 'CUBLAS_WORKSPACE_CONFIG',
            'HOST', 'HOSTNAME', 'SHELL', 'CUDA_VERSION', 'CUDA_PATH',
            *force_only_show_env_keys_extra_list,
        ]
        for key in list(env.keys()):
            if key not in only_show:
                env.pop(key, None)
                removed_keys.add(key)
    return dict(sorted(env.items()))
