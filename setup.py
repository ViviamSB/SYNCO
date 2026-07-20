from setuptools import setup, find_packages
from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

setup(
    name='synco',
    version='0.1.0',
    packages=find_packages(),
    description='SYNCO: A tool for computational and experimental synergy convergence for post-processing analysis',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Viviam Solangeli Bermudez',
    author_email='viviamsb@ntnu.no',
    url='https://github.com/ViviamSB/SYNCO',
    license='MIT',
    python_requires='>=3.9',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Topic :: Scientific/Engineering :: Medical/Pharma',
    ],
    keywords='synergy drug-response convergence DrugLogics',
    project_urls={
        'Documentation': 'https://github.com/ViviamSB/SYNCO#readme',
        'Bug Tracker': 'https://github.com/ViviamSB/SYNCO/issues',
        'Source Code': 'https://github.com/ViviamSB/SYNCO',
    },
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
        'PyYAML',
        'dash>=2.14',
        'dash-bootstrap-components>=1.5',
    ],
    entry_points={
        'console_scripts': [
            'synco = synco.cli:main',
        ],
    },
)