from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="customs-gun-object-detection",
    version="0.1",
    author="Muhammed Rasin",
    packages=find_packages(),
    install_requires=requirements,
)