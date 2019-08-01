import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="lifelong-forests",
    version="0.0.1",
    author="Hayden Helm",
    author_email="hh@jhu.edu",
    description="An implementation of lifelong forests",
    # long_description=long_description,
    # long_description_content_type="text/markdown",
    url="https://github.com/neurodata/lifelong-learning",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
