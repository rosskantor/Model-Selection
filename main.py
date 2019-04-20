# load libraries
import numpy as np
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

def run_code():
    #set random seed
    np.random.seed(0)

    #load data
    iris = datasets.load_iris()
    features = iris.data
    target = iris.target

    #create pipeline
    pipe = Pipeline([('classifier', RandomForestClassifier())])

    #create dictionaries with candidate learning algorithms and their hyperparameters
    search_space = [{'classifier':[LogisticRegression()],
                    'classifier__penalty': ['l1', 'l2'],
                    'classifier__C': np.logspace(0,4,10)},
                    {'classifier': [RandomForestClassifier()],
                    'classifier__n_estimators': [10,100,200, 300],
                    'classifier__max_depth': [1,2,3,4,5]}]

    # create grid search
    gridsearch = GridSearchCV(pipe, search_space, cv=5, verbose=0)

    #fit grid search
    best_model = gridsearch.fit(features, target)

    #return the best estimator

    return best_model, features

if __name__ == "__main__":
    run_code()
