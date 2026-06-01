from setuptools import setup, find_packages

setup(
    name="physics2finance",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.1.0",
        "torchvision>=0.16.0",
        "numpy>=1.24.0",
        "scipy>=1.11.0",
        "timm>=0.9.12",
        "einops>=0.7.0",
        "h5py>=3.10.0",
        "pandas>=2.1.0",
        "requests>=2.31.0",
        "arch>=6.3.0",
        "statsmodels>=0.14.0",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.8.0",
        "pyyaml>=6.0.1",
        "tqdm>=4.66.0",
        "loguru>=0.7.0",
    ],
    author="patriotsbreeze",
    description="Cross-Domain Latent Transfer from Physical Turbulence to Financial Market Microstructure",
    url="https://github.com/patriotsbreeze/Physics2Finance",
)
