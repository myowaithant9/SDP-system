# ***** Preprocessing_Stage Start *****
import pandas as pd
import openpyxl

def get_error_rate(pred, Y):
    return sum(pred != Y) / float(len(Y))

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
train_data, test_data = train_test_split(concat_data, test_size=0.28, shuffle=False) # random_state=42 # shuffle true --> 0.8951612903225806 # shuffle false --> 0.9274193548387096
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

from sklearn.svm import SVC
# classifier = SVC(kernel='rbf', random_state = 1)                #=> 0.9247  0.939873417721519
# classifier = svm.SVC(kernel='linear', C = 1.0)                  #=> 0.9247  0.939873417721519
# classifier = SVC(kernel='linear', C=1, random_state=0)          #=> 0.9247  0.939873417721519
# classifier = SVC(kernel='rbf', random_state=0, gamma=.01, C=1)  #=> 0.9247  0.939873417721519
# classifier = SVC(kernel='rbf', random_state=0, gamma=1, C=1)    #=> 0.9335443037974683
# classifier = SVC(kernel='rbf', random_state=0, gamma=10, C=1)   #=> 0.9335443037974683
# classifier = SVC(kernel='rbf', random_state=0, gamma=100, C=1)  #=> 0.9335443037974683

# classifier = SVC(kernel='rbf', random_state=0, gamma=.01, C=10)       #=> 0.939873417721519
# classifier = SVC(kernel='rbf', random_state=0, gamma=.01, C=1000)     #=> 0.939873417721519
# classifier = SVC(kernel='rbf', random_state=0, gamma=.01, C=10000)    #=> 0.9367088607594937
# classifier = SVC(kernel='rbf', random_state=0, gamma=.01, C=100000)   #=> 0.9208860759493671
classifier = SVC(kernel='rbf', random_state=0, gamma=.01, C=100000000)   #=> 0.9208860759493671


# C = 1.0
# # SVC with linear kernel
# classifier = svm.SVC(kernel='linear', C=C)            #=> 0.939873417721519
# # LinearSVC (linear kernel)
# classifier = svm.LinearSVC(C=C)                       #=> 0.939873417721519
# # SVC with RBF kernel
# classifier = svm.SVC(kernel='rbf', gamma=0.7, C=C)    #=> 0.939873417721519
# # SVC with polynomial (degree 3) kernel
# classifier = svm.SVC(kernel='poly', degree=3, C=C)    #=> 0.939873417721519


classifier.fit(feature_train_data,target_train_data)

Y_pred = classifier.predict(feature_test_data)
err_i = get_error_rate(Y_pred, target_test_data)
print('error rate-->', err_i)
# test_set["Predictions"] = Y_pred

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(target_test_data,Y_pred)
accuracy = float(cm.diagonal().sum())/len(target_test_data)
print("\nAccuracy Of SVM For The Given Dataset : ", accuracy)

# from sklearn.preprocessing import LabelEncoder
# le = LabelEncoder()
# target_train_data = le.fit_transform(target_train_data)

# from sklearn.svm import SVC
# classifier = SVC(kernel='rbf', random_state = 1)
# classifier.fit(feature_train_data,target_train_data)

# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.colors import ListedColormap
# plt.figure(figsize = (7,7))
# X_set, y_set = feature_train_data, target_train_data
# X1, X2 = np.meshgrid(np.arange(start = X_set[:, 9].min() - 1, stop = X_set[:, 9].max() + 1, step = 0.01), np.arange(start = X_set[:, 9].min() - 1, stop = X_set[:, 9].max() + 1, step = 0.01))
# plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape), alpha = 0.75, cmap = ListedColormap(('black', 'white')))
# plt.xlim(X1.min(), X1.max())
# plt.ylim(X2.min(), X2.max())
# for i, j in enumerate(np.unique(y_set)):
#     plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('red', 'orange'))(i), label = j)
#     plt.title('Apples Vs Oranges')
#     plt.xlabel('Weight In Grams')
#     plt.ylabel('Size in cm')
#     plt.legend()
#     plt.show()