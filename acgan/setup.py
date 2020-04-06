from setuptools import find_packages
from setuptools import setup

setup(
    name='acgan',
    version='0.1',
    description='Pytorch ACGAN acgan for Fashion MNIST.',
    install_requires=[
        'torchvision>=0.5.0',
        'torch>=1.4.0',
        'numpy>=1.18.2'
    ],
    packages=find_packages(),
    include_package_data=True,
    scripts=['train.py'],
    python_requires='>=3.7'
)
