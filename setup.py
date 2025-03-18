from setuptools import setup, find_packages

setup(
    name="spiq_project",
    version="0.1.0",
    description="Streaming Product Quantization pipeline for large-scale clustering of molecular data.",
    author="Alejandro Flores",
    author_email="afloresep01@gmail.com",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "Rdkit", 
        "pandas", 
        "tqmd", 
        "scikit-learn"
    ],
    entry_points={
        "console_scripts": [
            "spq=spq.main:main"
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)