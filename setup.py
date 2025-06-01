from setuptools import setup, find_packages

setup(
    name="temporal-anomaly-gnn",
    version="1.0.0",
    description="Temporal Graph Neural Networks for Real-time Fraud Detection",
    author="Mohammad Dindoost",
    author_email="md724@njit.edu",
    packages=find_packages(),
    install_requires=[
        "torch>=2.7.0",
        "torch-geometric>=2.6.1",
        "pandas>=1.5.0",
        "numpy>=1.21.0",
        "matplotlib>=3.5.0",
        "scikit-learn>=1.0.0",
        "networkx>=2.8.0",
    ],
    python_requires=">=3.10",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
