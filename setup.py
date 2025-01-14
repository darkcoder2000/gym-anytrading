from setuptools import setup, find_packages

setup(
    name='gym_anytrading_debug',
    version='1.2.1',
    packages=find_packages(),

    author='darkcoder',

    install_requires=[
        'gym>=0.12.5',
        'numpy>=1.16.4',
        'pandas>=0.24.2',
        'matplotlib>=3.1.1'
    ],

    package_data={
        'gym_anytrading_debug': ['datasets/data/*']
    }
)
