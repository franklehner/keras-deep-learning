"""
Setup
"""


from setuptools import setup, find_packages


setup(
    name='keras-deep-learning',
    version='1.0',
    author='Frank Lehner',
    author_email='fl@solute.de',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'scipy',
        'matplotlib',
        'Cython',
        'scikit-learn',
        'tensorflow',
        'theano',
        'keras',
        'h5py',
        'tables',
        'pillow',
        'opencv-python',
        'gensim',
        'click'
    ]
)
