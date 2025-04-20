"""Setup script for the SynthPix package."""

from setuptools import find_packages, setup

from src.hcnn import __version__ as version

setup(
    name="hcnn",
    version=version,
    author="Panagiotis Grontas and Antonio Terpin",
    description="Pi-net projection layer for Hard Constrained Neural Networks",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "jax==0.4.38",
        "tqdm>=4.67.1",
        "h5py>=3.13.0",
        "ruamel.yaml>=0.18.10",
        "imageio>=2.37.0",
        "matplotlib>=3.10.1",
        "optax==0.2.4",
        "flax==0.10.2",
        "cvxpy==1.6.0",
    ],
    extras_require={
        "dev": [
            "snowballstemmer==2.2.0",
            "pre_commit==4.0.1",
            "pytest==8.3.4",
        ],
        "cuda12": ["jax[cuda12_pip]==0.4.38"],
        "docs": [
            "Sphinx==7.4.7",
            "sphinx-copybutton==0.5.2",
            "sphinx-rtd-theme==2.0.0",
            "sphinx-tabs==3.4.7",
            "sphinx-togglebutton==0.3.2",
            "sphinxcontrib-applehelp==2.0.0",
            "sphinxcontrib-bibtex==2.6.2",
            "sphinxcontrib-devhelp==2.0.0",
            "sphinxcontrib-htmlhelp==2.1.0",
            "sphinxcontrib-jquery==4.1",
            "sphinxcontrib-jsmath==1.0.1",
            "sphinxcontrib-qthelp==2.0.0",
            "sphinxcontrib-serializinghtml==2.0.0",
        ],
    },
    python_requires=">=3.10",
)
