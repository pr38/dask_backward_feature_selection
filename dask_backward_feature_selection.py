import numpy as np

from sklearn.base import MetaEstimatorMixin, BaseEstimator
from sklearn.model_selection import cross_val_score
from sklearn.utils.validation import check_X_y, check_array
from sklearn.exceptions import NotFittedError

from itertools import combinations
from functools import partial

def features_scorer(X, y,estimator, feature_indexes, cv=None, scoring=None):
    score = cross_val_score(estimator,X[:,feature_indexes],y,cv=cv,scoring= scoring).mean()
    feature_indexes = feature_indexes

    return [score, feature_indexes]

class DaskBackwardFeatureSelector(MetaEstimatorMixin, BaseEstimator):
    def __init__(self, estimator, client,
                 k_features=1,
                 scoring=None,
                 cv=5,
                 ):

        self.estimator = estimator
        self.client = client
        self.k_features = k_features
        self.cv = cv
        self.scoring = scoring
        
        
        self.is_fitted = False
    
    def score_features(self, X, y):        
        outputs = []
        start = X.shape[1]
        
        feature_list_ =  range(X.shape[1])
        
        features_scorer_partial = partial(features_scorer, X,y,self.estimator,cv=self.cv,scoring=self.scoring)       
        
        for step in reversed(range(self.k_features,start,1)):
            
            combinations_ = list(combinations(feature_list_,step))
            
            
            futures = self.client.map(features_scorer_partial,combinations_)

            result = self.client.gather(futures)
    
            result_np = np.array(result)
                
            best_score_, feature_list_ = result_np[result_np[:,0].argsort()[::-1]][1].tolist()
        
            output = {}
            output['step'] = step
            output['feature_list_'] = feature_list_
            output['score'] = best_score_
            #output['results_np'] = result_np
            del result_np
            outputs.append(output)
            
        self.metric_dict = outputs
            
        outputs.sort(key=lambda a: a['score'],reverse = True)
            
        self.k_feature_idx_ = outputs[0]['feature_list_']
            
        self.metric_dict_ = outputs
        
    def check_is_fitted(self):
        if self.is_fitted != True:
            raise NotFittedError("""
            estimator is not fitted yet. Call 'fit' with appropriate arguments before using this estimator
            """)
            
    def fit(self,X,y):
        X, y = check_X_y(X, y)
        
        assert X.shape[1] > self.k_features

        self.score_features(X,y)
        self.estimator.fit(X[:,self.k_feature_idx_],y)
        
        self.is_fitted = True
        
        
        return self
    
    def transform(self,X,y=None):
        self.check_is_fitted()
        X = check_array(X)
        
        return X[:,self.k_feature_idx_], y
    
    def predict(self,X):
        self.check_is_fitted()
        X = check_array(X)
        
        return self.estimator.predict(X[:,self.k_feature_idx_])
