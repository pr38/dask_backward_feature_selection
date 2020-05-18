from setuptools import setup

setup(

    name='dask_backward_feature_selection', 
    version='0.0.2', 
    description='Backward step-wise feature selection using Dask, scikit-learn compatible',
    url='https://github.com/pr38/dask_backward_feature_selection', 
    install_requires=["numpy>=1.18.1","toolz>=0.9.0","scikit-learn>=0.20.1","dask>=2.6.0"],
    classifiers=[
    'Development Status :: 3 - Alpha',
    'Programming Language :: Python :: 3',
    ],
    packages=["dask_backward_feature_selection"],
    python_requires='>=3.5',
)
