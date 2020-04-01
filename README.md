## Dask Backward Feature Selection
Backward step-wise feature selection using Dask, scikit-learn compatible.
Scale out feature seletion useing distributed computing/Dask!

I created this due to the fact that mlxtend's SequentialFeatureSelector did not use joblib in a Dask compatable way.

Install
-------

> pip install git+https://github.com/pr38/dask_backward_feature_selection

Example Usage
-------
```python 
import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import load_boston,load_iris

from dask.distributed import Client, LocalCluster

from dask_backward_feature_selection import DaskBackwardFeatureSelector

#You should be useing Dask's yarn or kubernates cluster developments
#if you are going to be running this localy you are better off useing mlxtend's SequentialFeatureSelector 
cluster = LocalCluster(3)
client = Client(cluster)

le = LabelEncoder()
boston = load_boston()
X = boston['data']
y = le.fit_transform(boston['target'])

dfs = DaskBackwardFeatureSelector(DecisionTreeRegressor(),client)
#kwargs for DaskBackwardFeatureSelector are:
#k_features: the smallest combination of features DaskBackwardFeatureSelector will examine.
#cv: if "cv" is an int, it will refer to the number of  cross validation folds for each feature combination tested. 
#cv can also be a scikitlearn CV class.
#scoring: can be string (https://scikit-learn.org/stable/modules/generated/sklearn.metrics.get_scorer.html#sklearn.metrics.get_scorer)
#, or a scikitlearn scoring class.
#if scatter is true, each thread in the cluster will keep a copy of the training data and estimator.

dfs.fit(X,y)

#positions of top performing combination of features in X matrix.
dfs.k_feature_idx_

#we can treat DaskBackwardFeatureSelector as an estimator after training.
dfs.predict(X)


#also DaskBackwardFeatureSelector can act as transformer.
dfs.transform(X,y)

#finally we can examine the best perfomring feature combination for each step, for other use cases (ie:one-standard-error rule).
pd.DataFrame(dfs.metric_dict_ )
