# ***** Preprocessing_Stage Start *****
import pandas as pd
import openpyxl

dataset_path = 'PC3.csv'
dataset = pd.read_csv(dataset_path)
print('Dataset')
print(dataset)

indexNames = dataset[ dataset['DECISION_DENSITY'] == '?' ].index 
dataset.drop(indexNames , inplace=True)

col_count = 40
j = 0
print('\n')
print ("*** Remove Column Name ***")
for i in range(col_count):
    if dataset.iloc[:, i-j].nunique()==1:
        dropName = dataset.iloc[:, i-j].name
        print (dropName)
        dataset = dataset.drop(dropName, axis=1)
        j = j + 1
col_count = col_count - j

# print (dataset)
# df.to_excel("preprocess.xlsx")
# ***** Preprocessing_Stage End *****

array = dataset.values
feature_data = array[:,0:col_count]
target_data = array[:,col_count]

# ***** Discretization Start *****
from sklearn.preprocessing import KBinsDiscretizer

est = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='kmeans')  #uniform, quantile, kmeans
est.fit(feature_data)
discretize_data = est.transform(feature_data)
discretize_data = discretize_data.astype(int)
print('\n')
print ("*** Discretize Data ***")
print (discretize_data)
# ***** Discretization End *****

# ***** Feature Extraction Stage (MRMR) Start *****
import scipy.io
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn import svm
from skfeature.function.information_theoretical_based import MRMR

num_fea = 10 
feature_extraction = MRMR.mrmr(discretize_data, target_data, n_selected_features=num_fea)
print('\n')
print("*** Selected Feature ***")
print(feature_extraction)
# ***** Feature Extraction Stage (MRMR) End *****

# ***** Concat Stage Start *****
selected_data = discretize_data[:, [0,16,9,8,17,35,2,22,20,19]]

import numpy as np
concat_data = np.arange(12375).reshape(1125,11)
concat_data = np.concatenate((selected_data, target_data[:,None]), axis=1)
print('\n')
print('*** Concat Data ***')
print(concat_data)
print(concat_data.shape)
# ***** Concat Stage End *****

# ***** Split Data Stage Start *****
train_data, test_data = train_test_split(concat_data, test_size=0.33, shuffle=False) # random_state=42 # shuffle true --> 0.8951612903225806 # shuffle false --> 0.9274193548387096
print('\n')
print('*** Train Data ***')
print(train_data.shape)
print('*** Test Data ***')
print(test_data.shape)

feature_train_data = train_data[:,0:10]
target_train_data = train_data[:,10]

feature_test_data = test_data[:,0:10]
target_test_data = test_data[:,10]
# ***** Split Data Stage End *****

# Construct dataset
# X1, y1 = make_gaussian_quantiles(cov=2.,
#                                  n_samples=200, n_features=2,
#                                  n_classes=2, random_state=1)
# X2, y2 = make_gaussian_quantiles(mean=(3, 3), cov=1.5,
#                                  n_samples=300, n_features=2,
#                                  n_classes=2, random_state=1)
# X = np.concatenate((X1, X2))
# y = np.concatenate((y1, - y2 + 1))
from sklearn.ensemble import BaggingClassifier
# from sklearn.datasets import make_gaussian_quantiles
from sklearn.svm import SVC

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=5), algorithm="SAMME", n_estimators=400)

bdt.fit(feature_train_data, target_train_data)
# ***** Model Building Stage Start *****
# from sklearn.ensemble import AdaBoostClassifier
# from sklearn.datasets import make_classification
# clf = AdaBoostClassifier(n_estimators=100, random_state=0)
# a = clf.fit(feature_train_data, target_train_data)  
# print (a)
# y_predict = clf.predict(feature_test_data)
# print('y_predict')
# print(y_predict)
# correct = 0
# acc = accuracy_score(target_test_data, y_predict)
# correct = correct + acc
# print (correct)
# ***** Model Building Stage End *****



# import test_adaboost
# b = AdaBoost(n_estimators=100, random_state=0)
# c = b.fit(feature_train_data, target_train_data)
# print('Testing')
# print(b)

# print("feature train data -->")
# print(feature_train_data)
# print("feature train data end")

# # from sklearn import preprocessing
# # whole_data_x_scaled = preprocessing.scale(feature_train_data)
# # print("whole_data_x_scaled-->")
# # print(whole_data_x_scaled)
# # print("whole_data_x_scaled")

# from sklearn import svm
# from sklearn.model_selection import learning_curve
# from sklearn.model_selection import validation_curve
# from sklearn.model_selection import train_test_split
# from matplotlib import pyplot as plt
# # % matplotlib inline

# # Find better 'gamma' by default C value
# param_range = np.logspace(-2, 0, 20)
# print("param_range start")
# print(param_range)
# print("param_range end")

# train_scores, test_scores = validation_curve(
#     svm.SVC(C=0.6), feature_train_data, target_train_data, param_name="gamma", param_range=param_range,
#     cv=10, scoring="accuracy", n_jobs=1)
# print("train_scores")
# print(train_scores)
# print("test_scores")
# print(test_scores)

# train_scores_mean = np.mean(train_scores, axis=1)
# train_scores_std = np.std(train_scores, axis=1)
# test_scores_mean = np.mean(test_scores, axis=1)
# test_scores_std = np.std(test_scores, axis=1)

# plt.title("Validation Curve with SVM")
# plt.xlabel("$\gamma$")
# plt.ylabel("Score")
# plt.ylim(0.6, 1.1)
# lw = 2
# plt.semilogx(param_range, train_scores_mean, label="Training score",
#              color="darkorange", lw=lw)
# plt.fill_between(param_range, train_scores_mean - train_scores_std,
#                  train_scores_mean + train_scores_std, alpha=0.2,
#                  color="darkorange", lw=lw)
# plt.semilogx(param_range, test_scores_mean, label="Cross-validation score",
#              color="navy", lw=lw)
# plt.fill_between(param_range, test_scores_mean - test_scores_std,
#                  test_scores_mean + test_scores_std, alpha=0.2,
#                  color="navy", lw=lw)
# plt.legend(loc="best")
# plt.show()
# print(test_scores_mean)


# from sklearn.svm import SVC
# # from sklearn.preprocessing import StandardScaler
# # from sklearn.datasets import load_iris
# from sklearn.model_selection import StratifiedShuffleSplit
# from sklearn.model_selection import GridSearchCV

# C_range = [-2, 10, 13]
# gamma_range = [-9, 3, 13]
# param_grid = dict(gamma=gamma_range, C=C_range)
# cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
# grid = GridSearchCV(SVC(), param_grid=param_grid, cv=cv)
# print (grid)

# grid.fit(feature_train_data, target_train_data) 
# print("The best parameters are %s with a score of %0.2f"
#       % (grid.best_params_, grid.best_score_))
      

result = bdt.predict(feature_test_data)
print ("result")
print (result)


correct = 0
acc = accuracy_score(target_test_data, result)
print ("acctest")
print (acc)

correct = correct + acc
print (correct)


# clf = svm.SVC(C=1.0,gamma=0.1)

# # Fit all training data
# a = clf.fit(feature_train_data, target_train_data)
# # print ("a")
# # print (a)
# result = clf.predict(feature_test_data)
# # print('result')
# # print(result)
# correct = 0
# acc = accuracy_score(target_test_data, result)
# correct = correct + acc
# print (correct)