from setuptools import setup, find_packages

setup(
    name="tour_utils",
    author="PeterKucirek",
    version="0.1-dev",
    packages=find_packages(),

    requires=[
        'numpy',
        'pandas'
    ]
)
