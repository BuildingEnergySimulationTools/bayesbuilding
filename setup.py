"""BayesBuilding"""

from setuptools import setup, find_packages

# Get the long description from the README file
with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="bayesbuilding",
    version="0.1.0",
    description="Bayesian approach for building energy modeling",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/BuildingEnergySimulationTools/bayesbuilding",
    author="Nobatek/INEF4",
    author_email="bdurandestebe@nobatek.inef4.com",
    license="License :: OSI Approved :: BSD License",
    # keywords=[
    # ],
    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.10",
    install_requires=[
        "pymc>=5.10.3",
        "pandas>=1.3.4",
        "numpy>=1.26.3",
        "arviz>=0.17.0",
    ],
    packages=find_packages(exclude=["tests*"]),
    include_package_data=True,
)
