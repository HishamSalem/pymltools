from setuptools import setup, find_packages

setup(
    name="predykt",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'scikit-learn',
        'shap',
        'optbinning',
        'scipy',
        'statsmodels',
        'tqdm',
        'matplotlib',
        'seaborn',
        'joblib',
        'numba',
    ],
    extras_require={
        'salib': ['SALib'],
        'xgboost': ['xgboost'],
        'full': ['SALib', 'xgboost'],
    },
    author="Hisham Salem",
    author_email="hisham.salem@mail.mcgill.ca",
    description="A rigorous Python toolkit for predictive ML — feature analysis, interaction testing, and model robustness validation",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/HishamSalem/predykt",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)
