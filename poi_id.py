#!/usr/bin/python

import numpy as np
import pandas as pd
import sys
import pickle
import string

import matplotlib.pyplot as plt
from sklearn import neighbors
from sklearn import linear_model
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.cross_validation import KFold, train_test_split, StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.grid_search import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data, test_classifier


### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
features_list = ['poi','salary', 'deferral_payments', 'total_payments', 'loan_advances',
                 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value',
                 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive',
                 'restricted_stock', 'director_fees', 'to_messages', 'from_poi_to_this_person',
                 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi']

### Load the dictionary containing the dataset, create DataFrame from dictionary
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
    

### Task 2: Remove outliers
data_dict.pop('TOTAL', 0)
data_dict.pop('THE TRAVEL AGENCY IN THE PARK', 0) # from reading the pdf



### Task 3: Create new feature(s)

def compute_ratio(numerator, denominator, ratio_list):
'''
Defined a function to calculate the percentages and insert into data_dict
'''
    for name in data_dict:
        if data_dict[name][denominator] == 'NaN' or data_dict[name][denominator] == 0 \
        or data_dict[name][numerator] == 'NaN':
            data_dict[name][ratio_list] = 0
        if data_dict[name][denominator] != 'NaN' and data_dict[name][denominator] != 0 \
        and data_dict[name][numerator] != 'NaN':
            ratio = data_dict[name][numerator] / float(data_dict[name][denominator])
            data_dict[name][ratio_list] = ratio
        else:
            data_dict[name][ratio_list] = 'NaN'
    print "New feature", ratio_list, "is added to data_dict."

compute_ratio('from_this_person_to_poi', 'to_messages', 'ratio_to_poi')
compute_ratio('from_poi_to_this_person', 'from_messages', 'ratio_from_poi') 
compute_ratio('bonus', 'salary', 'ratio_bonus_to_salary')
features_list.extend(('ratio_to_poi', 'ratio_from_poi', 'ratio_bonus_to_salary'))


### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)




### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

### Feature selection by SelectKBest
skb = SelectKBest()
fit = skb.fit(features, labels)
temp = features_list[1:]
d = {'features': temp, 'scores': fit.scores_}
scores_df = pd.DataFrame(d).sort_values(by='scores', ascending=False)
print scores_df

### Select the cutoff point where feature has the smallest marginal benefit to the model
scores_df.plot.bar(x='features')

### Selected top features according to the fit scores
features_selected = ['poi', 'exercised_stock_options', 'total_stock_value', 'bonus', 'salary']
data = featureFormat(my_dataset, features_selected, remove_NaN=True, remove_all_zeroes=True, sort_keys=True, remove_any_zeroes=True)
labels, features = targetFeatureSplit(data)
sss = StratifiedShuffleSplit(labels, n_iter=100, train_size=0.75, random_state=46)

### Try decision tree
tree = DecisionTreeClassifier()
params = {'min_samples_split': range(5, 41, 5),
         'class_weight': ['balanced', {True: 10, False: 1}, {True: 8, False: 1}, {True: 6, False: 1}],
         'splitter': ['random', 'best']
         }
gs = GridSearchCV(tree, param_grid=params, scoring='precision', cv=sss)
gs.fit(features, labels)
tree_estimator = gs.best_estimator_
print "Best Estimator: ", tree_estimator
print "Best Params: ", gs.best_params_
test_classifier(tree_estimator, my_dataset, features_list, folds = 1000)


### Try SVM
svc = SVC()
scaler = MinMaxScaler()
pipe = Pipeline([('scaler', scaler), ('svc', svc)])
params = {'svc__kernel':['rbf', 'linear'], 
          'svc__degree': range(1, 3),
          'svc__C': range(8, 25, 2),
          'svc__class_weight': [{True: 14, False: 1}, {True: 12, False: 1}, {True: 10, False: 1}]
          }
gs = GridSearchCV(pipe, param_grid=params, scoring='precision', cv=sss)
gs.fit(features, labels)
svc_estimator = gs.best_estimator_
print "Best Estimator: ", svc_estimator
print "Best Params: ", gs.best_params_
test_classifier(svc_estimator, my_dataset, features_selected, folds = 1000)


### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

scaler = MinMaxScaler()
knn = KNeighborsClassifier()
pipeline = Pipeline([('scaler', scaler), ('knn', knn)])
params = {'knn__n_neighbors': range(1, 10),
          'knn__weights': ['uniform', 'distance'],
          'knn__algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
          }
gs = GridSearchCV(pipeline, param_grid=params, scoring='f1', cv=sss)
gs.fit(features, labels)
knn_estimator = gs.best_estimator_
print "Best Estimator: ", knn_estimator
print "Best Params: ", gs.best_params_
test_classifier(knn_estimator, my_dataset, features_selected, folds = 1000)


### Re-testing the best knn classifier
clf = KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski', metric_params=None, n_jobs=1, n_neighbors=1, p=2, weights='uniform')
test_classifier(clf, my_dataset, features_selected, folds = 1000)



### Task 5.5: Test one of the newly implemented features

### Adding the new feature ratio_bonus_to_salary to the data
features_test = ['poi', 'exercised_stock_options', 'total_stock_value', 'bonus', 'salary', 'ratio_bonus_to_salary']

clf_test = KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski', metric_params=None, n_jobs=1, n_neighbors=1, p=2, weights='uniform')
test_classifier(clf_test, my_dataset, features_test, folds = 1000)


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_selected)
