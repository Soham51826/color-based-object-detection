"""
setup.py
========
Package build configuration for the Color-Based Object Detection System.

Install in editable/development mode:
    pip install -e .

Install for production:
    pip install .
"""

from pathlib import Path
from setuptools import setup, find_packages

HERE = Path(__file__).parent
LONG_DESCRIPTION = (HERE / "README.md").read_text(encoding="utf-8")

setup(
    name="color-object-detection",
    version="1.0.0",
    author="Open Source Contributors",
    description=(
        "Production-ready color-based object detection and real-time tracking "
        "system in OpenCV"
    ),
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    url="https://github.com/Soham51826/color-based-object-detection",
    license="MIT",
    packages=find_packages(exclude=["tests*"]),
    python_requires=">=3.9",
    install_requires=[
        "opencv-python>=4.8.0",
        "numpy>=1.24.0",
        "matplotlib>=3.7.0",
        "Pillow>=10.0.0",
        "scipy>=1.11.0",
        "customtkinter>=5.2.0",
        "python-dotenv>=1.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.0.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "color-detect=src.gui_app:main",
        ]
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Education",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
    keywords="computer-vision object-detection HSV color-tracking OpenCV",
)
