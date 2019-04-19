# Model-Selection
The ability to train a model is immensely import.  Selecting the correct features, splitting datasets into train and test sets, running models, and validating is time consuming.  After all the work, what if you selected the wrong model.  Enter GridSearchCV.
This code uses a feature I did not know existed: the ability to feed multiple classifiers to the same grid search object.  The gridsearch object return the best model allowing end users to predict and extract model attributes.
