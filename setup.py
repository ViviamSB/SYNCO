from setuptools import setup, find_packages

setup(
    name='SYNCO',
    version='0.1',
    packages=find_packages(),
    description='SYNCO: A tool for computational and experimental synergy convergence for post-processing analysis',
    author='Viviam Solangeli Bermudez',
    author_email='viviamsb@ntnu.no',
    url='https://github.com/ViviamSB/SYNCO',
    install_requires=[
        'numpy',
        'pandas',
        'scikit-learn',
        'matplotlib',
        'seaborn',
        'plotly',
        'kaleido',
        'streamlit>=1.28.0',
        'streamlit-option-menu>=0.3.0',
    ],
    entry_points={
        'console_scripts': [
            'synco = synco.cli:main',
        ],
    },
)