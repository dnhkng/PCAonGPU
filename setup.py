from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="PCAonGPU",
    version="0.1", 
    author="Your Name",
    author_email="dnhkng@gmail.com",
    description="A GPU-based Incremental PCA implementation.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/dnhkng/PCAonGPU",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.10',
    install_requires=[
        'torch>=2.1.0'
    ],
    extras_require={
        'test': [
            'pytest>=1.3.2',
            'scikit-learn>=1.3.2',
            'numpy>=1.3.2',
        ],
    },
)
