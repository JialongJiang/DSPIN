from setuptools import find_packages, setup
import os

install_requires = [
    "anndata",
    "matplotlib",
    "scanpy",
    "tqdm",
    "igraph",
    "leidenalg",
]

# # Replace with your package's test requirements
# test_requires = [
#     # Example: "pytest", "pytest-cov"
# ]

# # Replace with your package's documentation requirements
# doc_requires = [
#     # Example: "sphinx"
# ]

# Managing your version (adjust accordingly)
version = "1.2.0"  # You can manage versioning in a more sophisticated way if needed

# Long description (usually a README file)
readme = open("README_PyPI.md").read()  # Adjust the file name if necessary

setup(
    name="dspin",
    version=version,
    description="Regulatory network models from single-cell perturbation profiling",
    author="Jialong Jiang",
    author_email="jiangjl@caltech.edu",
    packages=find_packages(),
    license="Apache-2.0 license",
    python_requires=">=3.6",  # Adjust depending on your compatibility
    install_requires=install_requires,
    # extras_require={"test": test_requires, "doc": doc_requires},
    # Adjust if you're using a different test suite
    test_suite="nose2.collector.collector",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/JialongJiang/DSPIN",  # Replace with your repository URL
    # download_url="https://github.com/JialongJiang/DSPIN/archive/v{}.tar.gz".format(version),
    keywords=[
        "network inference",
        "single cell",
        "transcriptomics"
        # Add more relevant keywords
    ],
    classifiers=[
        # Update classifiers to match your project
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: Apache Software License",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Scientific/Engineering :: Physics",
        # Add more classifiers as relevant
    ],
)
