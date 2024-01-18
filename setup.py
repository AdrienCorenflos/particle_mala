"""
Setup script for gradient_csmc package.
"""

from setuptools import find_packages
from setuptools import setup
from setuptools.command.install import install as InstallCommandBase

REQUIRED_PACKAGES = [
    'chex',
    'jax[cpu]',
    'matplotlib',
    'seaborn',
    'statsmodels',
    'tensorflow_probability',
    'tqdm'
]

with open('README.md', 'r') as fh:
    long_description = fh.read()

setup(
    name="gradient-csmc",
    version="0.1",
    description="Gradient-informed conditional sequential Monte Carlo. Written in JAX",
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Adrien Corenflos',
    author_email='adrien.corenflos@gmail.com',
    url='http://github.com/AdrienCorenflos/mala_csmc',
    license='MIT License',
    packages=find_packages(),
    python_requires='>=3.9',
    install_requires=REQUIRED_PACKAGES,
    # Add in any packaged data.
    include_package_data=True,
    zip_safe=False,
    cmdclass={
        'pip_pkg': InstallCommandBase,
    },
    keywords='jax statistics bayesian state-space-models monte-carlo sequential-monte-carlo particle-filter '
             'particle-smoother particle-mcmc',
)
