# 2018-April
# github.com/ludlows
# Canny Line (a toolbox for line segments extraction)

from setuptools import find_packages
from setuptools import setup, Extension

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="cannyline",
    version="0.0.0",
    author="ludlows",
    description="Canny Line (a toolbox for line segments extraction)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ludlows/CannyLine",
    packages=find_packages(),
    package_data={'cannyline':["*.py"]},
    ext_package='cannyline',
    setup_requires=['setuptools', 'opencv-python', 'numpy'],
    classifiers=[
        "Programming Language :: Python",
        "Operating System :: OS Independent",
    ]
)