import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="gspy",
    version="0.0.1",
    author="Guilherme Boaviagem",
    author_email="guilherme.boaviagem@gmail.com",
    description="Utilities for Graph Signal Processing",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/gboaviagem/GSPy",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
