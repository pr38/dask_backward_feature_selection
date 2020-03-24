## Dask Backward Feature Selection
Backward step-wise feature selection using Dask, scikit-learn compatible

I created this due to the fact that mlxtend's SequentialFeatureSelector did not use joblib in a Dask compatable way.


Install
-------
via  ``pip``:

::

   # Install with conda
   $ conda install dask-searchcv -c conda-forge

   # Install with pip
   $ pip install git+https://github.com/prez38/dask_backward_feature_selection


