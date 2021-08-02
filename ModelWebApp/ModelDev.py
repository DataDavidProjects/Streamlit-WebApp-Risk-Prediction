import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


df = pd.read_csv('Bank_Personal_Loan_Modelling.csv')


from sklearn.datasets import make_classification
X,y = make_classification(n_samples=10000, n_features=3, n_informative=3,
                          n_redundant=0, n_repeated=0, n_classes=2,
                          n_clusters_per_class=2,
                          class_sep=1.5,
                          flip_y=0,weights=[0.5,0.5])

features = pd.DataFrame(X)
target = pd.Series(y)



from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

model= RandomForestClassifier(max_depth = 5, random_state = 0, n_estimators = 100)
model.fit(features,target)


import pickle
pickle.dump(model, open('model', 'wb'))
model = pickle.load(open('model', 'rb'))


def predict_model(features):
    predictions=model.predict(features)
    return predictions

print(predict_model(X))


features.to_csv('test.csv',index=False)


features_names = [0,1,2]
print(features_names)
# Importing the module for LimeTabularExplainer


import lime.lime_tabular
explainer = lime.lime_tabular.LimeTabularExplainer(features.values,
                                                   feature_names=features_names,
                                                   class_names=[0, 1],
                                                   feature_selection="lasso_path", discretize_continuous=True,
                                                   discretizer="quartile", verbose=True, mode='classification'
                                                   )
exp = explainer.explain_instance(features.iloc[0], model.predict_proba,)
#find a plot option
print(exp.as_list())