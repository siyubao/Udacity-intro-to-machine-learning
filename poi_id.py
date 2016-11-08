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

'''
Defined a function to calculate the percentages and insert into data_dict
'''
def compute_ratio(numerator, denominator, ratio_list):
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

def clf_accuracy(classifier):
    clf = classifier()
    clf.fit(features, labels)
    pred = clf.predict(features)
    return classifier, "Accuracy:", "%.3f" % accuracy_score(pred, labels)

### All classifiers are overfit
print clf_accuracy(GaussianNB)
print clf_accuracy(SVC)
print clf_accuracy(DecisionTreeClassifier)
print clf_accuracy(neighbors.KNeighborsClassifier)
print clf_accuracy(AdaBoostClassifier)
print clf_accuracy(RandomForestClassifier)


### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

### Fill all missing values with zero, so they can be discounted by MinMaxScaler
df = pd.DataFrame.from_dict(data_dict, orient='index', dtype=np.float)
df = df.replace('NaN', 0)

### Remove string data in column email_address
del df['email_address']

### Transform all the features into floating points and seperate the label
features_df = df.drop(['poi'], axis=1)
labels_df = df['poi']

scaler = MinMaxScaler()
features_rescaled = scaler.fit_transform(features_df)

### Update features array, turn labels into array
features = features_rescaled
labels = labels_df.as_matrix().astype(int)


### Use SelectKBest to find the cut-off point in the number of features selected
### Chi-square scores work better with normalized data
### Hard-coded the features according to the results in the write-up because SelectKBest produces different outcomes

#skb = SelectKBest(chi2, k='all')
#fit = skb.fit(features, labels)

#temp = features_list[1:]
#d = {'features': temp, 'scores': fit.scores_}
#chi2_df = pd.DataFrame(d).sort_values(by='scores', ascending=False)
#chi2_df

### Visualize the score of each data, select cut-off point at the sharpest decline
#chi2_df.plot.bar(x='features')

features_selected = ['poi', 'bonus', 'long_term_incentive', 'exercised_stock_options',
                     'restricted_stock_deferred']


### Validation

### Extract features and labels from dataset
data = featureFormat(my_dataset, features_selected, sort_keys = True, remove_all_zeroes = True, remove_NaN = True)
labels, features = targetFeatureSplit(data)

### Use StratifiedShuffleSplit to maximize the 
sss = StratifiedShuffleSplit(labels, n_iter=100, test_size=0.25, train_size=0.6, random_state=46)
for train_idx, test_idx in sss: 
        features_train = []
        features_test  = []
        labels_train   = []
        labels_test    = []
        for ii in train_idx:
            features_train.append(features[ii] )
            labels_train.append(labels[ii] )
        for jj in test_idx:
            features_test.append(features[jj] )
            labels_test.append(labels[jj] )

### Try DecisionTree, no need to perform feature scaling
clf = DecisionTreeClassifier()
params = {'min_samples_split': range(2, 11),
         'class_weight': ['balanced', {True: 12, False: 1}, {True: 10, False: 1}, {True: 8, False: 1}],
         'splitter': ['random', 'best']
         }
clf_gs = GridSearchCV(clf, param_grid=params, scoring='f1')
clf_gs.fit(features_train, labels_train)
clf_estimator = clf_gs.best_estimator_
clf_params = clf_gs.best_params_
print "Best Estimator: ", clf_estimator
print "Best Params: ", clf_params
test_classifier(clf_estimator, my_dataset, features_selected, folds = 1000)




### Random Forest Classifier
forest = RandomForestClassifier()
params = {'n_estimators': range(1, 10),
          'criterion': ['gini', 'entropy'],
          'min_samples_split': range(1, 6, 2),
          'class_weight': ['balanced', {True: 12, False: 1}, {True: 10, False: 1}, {True: 8, False: 1}]
          }
forest_gs = GridSearchCV(forest, param_grid=params, scoring='f1')
forest_gs.fit(features, labels)
forest_estimator = forest_gs.best_estimator_
forest_params = forest_gs.best_params_
print "Best Estimator: ", forest_estimator
print "Best Params: ", forest_params
test_classifier(forest_estimator, my_dataset, features_selected, folds = 1000)




### K Neighbors Classifier
scaler = MinMaxScaler()
knn = KNeighborsClassifier()
pipeline = Pipeline([('scaler', scaler), ('knn', knn)])
params = {'knn__n_neighbors': range(1, 10),
          'knn__weights': ['uniform', 'distance'],
          'knn__algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
          }
knn_gs = GridSearchCV(pipeline, param_grid=params, scoring='recall')
knn_gs.fit(features, labels)
knn_estimator = knn_gs.best_estimator_
knn_params = knn_gs.best_params_
print "Best Estimator: ", knn_estimator
print "Best Params: ", knn_params
test_classifier(knn_estimator, my_dataset, features_selected, folds = 1000)


### Try SVM, adjusted multiple times to find the best range of the C parameter
svc = SVC()
scaler = MinMaxScaler()
pipe = Pipeline([('scaler', scaler), ('svc', svc)])
params = {'svc__kernel':['rbf', 'linear'], 
          'svc__degree': range(1, 3),
          'svc__C': range(8, 25, 2),
          'svc__class_weight': [{True: 14, False: 1}, {True: 12, False: 1}, {True: 10, False: 1}]
          }
svc_gs = GridSearchCV(pipe, param_grid=params, scoring='precision', cv=sss)
svc_gs.fit(features, labels)
svc_estimator = svc_gs.best_estimator_
svc_params = svc_gs.best_params_
print "Best Estimator: ", svc_estimator
print "Best Params: ", svc_params
test_classifier(svc_estimator, my_dataset, features_selected, folds = 1000)



### Test one of the newly implemented features
### Ratio-bonus_to_salary
features_selected.append('ratio_bonus_to_salary')

data = featureFormat(my_dataset, features_selected, sort_keys = True, remove_all_zeroes = True, remove_NaN = True)
labels, features = targetFeatureSplit(data)
sss = StratifiedShuffleSplit(labels, n_iter=100, test_size=0.25, train_size=0.6, random_state=46)
for train_idx, test_idx in sss: 
        features_train = []
        features_test  = []
        labels_train   = []
        labels_test    = []
        for ii in train_idx:
            features_train.append(features[ii] )
            labels_train.append(labels[ii] )
        for jj in test_idx:
            features_test.append(features[jj] )
            labels_test.append(labels[jj] )
test = DecisionTreeClassifier()
params = {'min_samples_split': range(2, 11),
         'class_weight': ['balanced', {True: 12, False: 1}, {True: 10, False: 1}, {True: 8, False: 1}],
         'splitter': ['random', 'best']
         }
test_gs = GridSearchCV(clf, param_grid=params, scoring='f1')
test_gs.fit(features_train, labels_train)
test_estimator = test_gs.best_estimator_
test_params = test_gs.best_params_
print "Best Estimator: ", test_estimator
print "Best Params: ", test_params
test_classifier(test_estimator, my_dataset, features_selected, folds = 1000)

### With the new feature ratio_salary_to_bonus, the DecisionTreeClassifier no longer satisfies the 0.3 requirements.




### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf_estimator, my_dataset, features_list)

