"""Setup script for OCINR package."""

from setuptools import setup, find_packages
from pathlib import Path

# Read the contents of README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

# Read requirements
requirements = []
with open("requirements.txt") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

dev_requirements = []
with open("requirements-dev.txt") as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith("#") and not line.startswith("-r"):
            dev_requirements.append(line)

setup(
    name="ocinr",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Optimal Control Implicit Neural Representations for scientific data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/OCINR",
    project_urls={
        "Bug Tracker": "https://github.com/yourusername/OCINR/issues",
        "Documentation": "https://github.com/yourusername/OCINR",
        "Source Code": "https://github.com/yourusername/OCINR",
    },
    packages=find_packages(exclude=["tests", "tests.*", "notebooks", "scripts"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={
        "dev": dev_requirements,
    },
    entry_points={
        "console_scripts": [
            "ocinr-train=src.train:main",
            "ocinr-eval=src.eval:main",
        ],
    },
)

