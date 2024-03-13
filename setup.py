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
version = "0.1.1"  # You can manage versioning in a more sophisticated way if needed

# Long description (usually a README file)
readme = open("README.md").read()  # Adjust the file name if necessary

setup(
    name="dspin",
    version=version,
    description="Short description of dspin",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    license="Your chosen license",
    python_requires=">=3.6",  # Adjust depending on your compatibility
    install_requires=install_requires,
    # extras_require={"test": test_requires, "doc": doc_requires},
    test_suite="nose2.collector.collector",  # Adjust if you're using a different test suite
    long_description=readme,
    url="https://github.com/YingyGong/dspin",  # Replace with your repository URL
    download_url="https://github.com/YingyGong/dspin/archive/v{}.tar.gz".format(
        version
    ),
    keywords=[
        "keyword1",
        "keyword2",
        # Add more relevant keywords
    ],
    classifiers=[
        # Update classifiers to match your project
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        # Add more classifiers as relevant
    ],
)
