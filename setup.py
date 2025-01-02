from setuptools import setup, find_packages

setup(
    name="ml-feature-toolkit",
    version="0.0.1",
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'scikit-learn',
        'xgboost',
        'shap',
        'optbinning',
        'SALib',
        'scipy',
        'statsmodels',
        'tqdm',
        'matplotlib',
        'seaborn',
        'joblib'
    ],
    author="Hisham Salem",
    author_email="hisham.salem@mail.mcgill.ca",
    description="A package for analyzing feature interactions in machine learning models",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/HishamSalem/pymltools",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
