from setuptools import setup, find_packages

setup(
    name="unet_compare",
    version="0.1",
    url="https://github.com/DuncanB6/UofC2022",
    author="Duncan Boyd",
    author_email="duncan@wapta.ca",
    description="Contain a complex unet, a real unet, and the functions necessary to run them.",
    packages=find_packages(),
    # install_requires=['numpy >= 1.11.1', 'matplotlib >= 1.5.1'],
)
