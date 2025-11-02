"""
Package setup
"""
from setuptools import setup, find_packages

setup(
    name="agriculture-disease-detection",
    version="1.0.0",
    description="AI-powered crop disease detection for Indian farmers",
    author="ByteForge Titans",
    packages=find_packages(),
    install_requires=[
        line.strip()
        for line in open('requirements.txt').readlines()
        if line.strip() and not line.startswith('#')
    ],
    python_requires='>=3.10',
)
