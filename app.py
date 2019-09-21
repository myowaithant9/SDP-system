# for pc1 svm

from main import preprocess, discretize, feature_extract, concat, generic_clf, adaboost_clf
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# Predefined value
datactrl = 0
modelctrl = 1   #1 --> Ada, 0 --> SVM
###################
if datactrl == "PC1":
    dataset_path = 'MDP csv/PC01.csv'
if datactrl == "PC2":
    dataset_path = 'MDP csv/PC02.csv'
if datactrl == "PC3":
    dataset_path = 'MDP csv/PC03.csv'
if datactrl == "PC4":
    dataset_path = 'MDP csv/PC04.csv'
if datactrl == "PC5":
    dataset_path = 'MDP csv/PC05.csv'
print("dataset_path-->", dataset_path)
return dataset_path

feature_data, target_data = preprocess(dataset_path, datactrl)
print('feature_data')
print(feature_data)
print(feature_data.shape)

print('target_data')
print(target_data)
print(target_data.shape)

discretize_data = discretize(feature_data)
print('\n')
print ("*** Discretize Data ***")
print (discretize_data)
print(discretize_data.shape)

concat_data = 0
if modelctrl == 1 :
    num_fea = 10
    feature_extraction = feature_extract(discretize_data, target_data, num_fea)
    print('\n')
    print("*** Selected Feature ***")
    print(feature_extraction)

    if datactrl == 1:
        selected_data = discretize_data[:, [36, 29, 16,  3, 32,  9, 19,  4, 20, 22]]  #=> transform manual to auto
    if datactrl == 2:
        selected_data = discretize_data[:, [3, 28, 14, 13,  8,  4, 21, 25, 16, 15]]  #=> transform manual to auto
    if datactrl == 3:
        selected_data = discretize_data[:, [0, 16, 9,  8, 17, 35,  2, 22, 20, 19]]  #=> transform manual to auto
    if datactrl == 4:
        selected_data = discretize_data[:, [35, 25,  3, 16, 17,  9,  7, 32, 10, 29]]  #=> transform manual to auto
    if datactrl == 5:
        selected_data = discretize_data[:, [34, 18,  1,  3,  2,  7, 19, 15, 26, 12]]  #=> transform manual to auto

    concat_data = concat(selected_data, target_data)

else:
    concat_data = concat(discretize_data, target_data)

print('\n')
print('*** Concat Data ***')
print(concat_data)
print(concat_data.shape)

train_data, test_data = train_test_split(concat_data, test_size=0.3, shuffle=False) # random_state=42 # shuffle true --> 0.8951612903225806 # shuffle false --> 0.9274193548387096
print('\n')
print('*** Train Data ***')
print(train_data.shape)
print('*** Test Data ***')
print(test_data.shape)

## for 0 case, SVM
if modelctrl == 0:
    X_train = train_data[:,0:37]
    Y_train = train_data[:,37].astype('int')

    X_test = test_data[:,0:37]
    Y_test = test_data[:,37].astype('int')

if modelctrl == 1:
    X_train = train_data[:,0:10]
    Y_train = train_data[:,10].astype('int')

    X_test = test_data[:,0:10]
    Y_test = test_data[:,10].astype('int')

clf_tree = SVC(kernel='rbf', random_state=0, gamma=.01, C=10000)

if modelctrl == 0:
    er_tree = generic_clf(Y_train, X_train, Y_test, X_test, clf_tree)
    print('clf_tree', clf_tree)
    print('er_tree', er_tree)
    er_train, er_test = [er_tree[0]], [er_tree[1]]
    print('er_train', er_train)
    print('er_test', er_test)

if modelctrl == 1:
    if (datactrl == 1):
        x_range = range(0, 9, 1)
    if (datactrl == 2):
        x_range = range(0, 16, 1)
    if (datactrl == 3):
        x_range = range(0, 6, 1)
    if (datactrl == 4):
        x_range = range(0, 2, 1)
    if (datactrl == 5):
        x_range = range(0, 8, 1)
    for i in x_range:    
        er_i = adaboost_clf(Y_train, X_train, Y_test, X_test, i, clf_tree)
        print('')
        print('er_i'+ str(i) +'-->', er_i)

        # er_train.append(er_i[0])
        # er_test.append(er_i[1])
