from setuptools import setup, find_packages

setup(
    name="odor_space_sampling",
    version="0.1.0",
    description="Sampling methods to generate lists of odorants that span odor space",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.8",
    install_requires=[
        "numpy",
        "pandas",
        "requests",
        "tqdm",
        "rdkit",
        "matplotlib",
        "scikit-learn",
        "scipy",
    ],
)
