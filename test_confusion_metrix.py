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
print ('feature list')
print (dataset.columns)
dataset.to_excel("preprocess.xlsx")
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
print(discretize_data.shape)
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

# ***** Model Building Stage Start *****
from sklearn.ensemble import AdaBoostClassifier
from sklearn.datasets import make_classification
clf = AdaBoostClassifier(n_estimators=100, random_state=0)
a = clf.fit(feature_train_data, target_train_data)  
print (a)
y_predict = clf.predict(feature_test_data)
print('y_predict')
print(y_predict)
correct = 0
acc = accuracy_score(target_test_data, y_predict)
correct = correct + acc
print (correct)
# ***** Model Building Stage End *****

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(target_test_data,y_predict)
accuracy = float(cm.diagonal().sum())/len(target_test_data)
print("\nAccuracy Of AdaBoost For The Given Dataset : ", accuracy)