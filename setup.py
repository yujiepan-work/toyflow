from setuptools import setup, find_packages


setup(
    author="yujiepan",
    author_email='yujiepan@no-email.example.com',
    python_requires='>=3.8',
    description="Lightweight flow execution tool.",
    license="MIT license",
    keywords='toytools',
    name='toytools',
    packages=find_packages(include=['toytools', 'toytools.*']),
    url='https://github.com/yujiepan-work/toytools',
    version='0.1.0',
)