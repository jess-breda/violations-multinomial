from setuptools import setup, find_packages
from pathlib import Path

# Path to the directory of setup.py
setup_dir = Path(__file__).parent

# Path to requirements.txt (two directories up)
requirements_path = setup_dir.parent / "requirements.txt"

# Read the contents of the requirements file
with open(requirements_path) as f:
    requirements = f.read().splitlines()

setup(
    name="violmulti",
    version="0.2.0",
    packages=find_packages(),
    install_requires=requirements,
    description="Multinomial modeling of animal choice data (L,R,Violation)",
    author="Jess Breda",
    author_email="jbreda@princeton.edu",
)
