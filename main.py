import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix

""" Discretization SCRIPT ============================================================="""
def main_preprocess(dataset_path, datactrl):
    col_count = 37            

    dataset = pd.read_csv(dataset_path)
    np.set_printoptions(formatter={'float_kind':'{:0f}'.format})

    print('Dataset')
    print(dataset)
    if datactrl != 1:
        indexNames = dataset[ dataset['DECISION_DENSITY'] == '?' ].index 
        dataset.drop(indexNames , inplace=True)

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

    array = dataset.values
    feature_data = array[:,0:col_count]
    target_data = array[:,col_count]
    return dataset, feature_data, target_data

""" Discretization SCRIPT ============================================================="""
def main_discretize(feature_data):
    from sklearn.preprocessing import KBinsDiscretizer

    est = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='kmeans')  #uniform, quantile, kmeans
    est.fit(feature_data)
    print("est.fit", est.fit(feature_data))
    discretize_data = est.transform(feature_data)
    print("est.transform", est.transform(feature_data))

    discretize_data = discretize_data.astype(int)
    return discretize_data

""" Feature Extraction SCRIPT ============================================================="""
def feature_extract(discretize_data, target_data, num_fea):
    import scipy.io
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import cross_validate
    from sklearn.model_selection import train_test_split
    from sklearn import svm
    from skfeature.function.information_theoretical_based import MRMR

    feature_extraction = MRMR.mrmr(discretize_data, target_data, n_selected_features=num_fea)
    return feature_extraction

def concat(selected_data, target_data):
    import numpy as np
    concat_data = np.arange(8349).reshape(759,11)
    concat_data = np.concatenate((selected_data, target_data[:,None]), axis=1)
    return concat_data

""" HELPER FUNCTION: GENERIC CLASSIFIER ====================================="""
def generic_clf(Y_train, X_train, Y_test, X_test, clf):
    clf.fit(X_train,Y_train)
    pred_train = clf.predict(X_train)
    pred_test = clf.predict(X_test)
#     print('pred_train for generic', pred_train)
#     print('pred_test for generic', pred_test)
#     print('PC01precision_recall_fscore_support avg none: {}', precision_recall_fscore_support(Y_test, pred_test, average=None))
#     print('Accuracy for Generic PC01', accuracy_score(Y_test, pred_test))
    
#     cm = confusion_matrix(Y_test, pred_test)
#     print("\nConfusion Matrix PC01", cm)

    return pred_train, pred_test

#     return get_error_rate(pred_train, Y_train), \
#            get_error_rate(pred_test, Y_test)
    
""" HELPER FUNCTION: GET ERROR RATE ========================================="""
def get_error_rate(pred, Y):
    return sum(pred != Y) / float(len(Y))

# def adaboost_clf_pre(Y_train, X_train, Y_test, X_test, M, clf):
#     acc_ada = 0
#     prf_ada = 0
#     cm_ada = 0    

#     def adaboost_clf(Y_train, X_train, Y_test, X_test, M, clf):

#         n_train, n_test = len(X_train), len(X_test)
#         # Initialize weights
#         w = np.ones(n_train) / n_train
#         # print('w first one', w)
#         pred_train, pred_test = [np.zeros(n_train), np.zeros(n_test)]

#         for i in range(M):
#                 global acc_ada 
#                 global prf_ada
#                 global cm_ada
#                 clf.fit(X_train, Y_train, sample_weight = w)
#                 pred_train_i = clf.predict(X_train)
#                 pred_test_i = clf.predict(X_test)

#                 prf_ada = precision_recall_fscore_support(Y_test, pred_test_i, average=None)
#                 print(str(i) + 'precision_recall_fscore_support avg none: {}', prf_ada)
                
#                 acc_ada = accuracy_score(Y_test, pred_test_i)
#                 print('Accuracy for'+ str(i) +'-->', acc_ada)
                
#                 cm_ada = confusion_matrix(Y_test, pred_test_i)
#                 print("\nConfusion Matrix", cm_ada)

#                 # Indicator function
#                 miss = [int(x) for x in (pred_train_i != Y_train)]
#                 # Equivalent with 1/-1 to update weights
#                 miss2 = [x if x==1 else -1 for x in miss]
#                 # Error
#                 err_m = np.dot(w,miss) / sum(w)

#                 # Alpha
#                 alpha_m = 0.5 * np.log( (1 - err_m) / float(err_m))

#                 # New weights
#                 w = np.multiply(w, np.exp([float(x) * alpha_m for x in miss2]))

#                 # Add to prediction
#                 pred_train = [sum(x) for x in zip(pred_train, 
#                                                 [x * alpha_m for x in pred_train_i])]
#                 pred_test = [sum(x) for x in zip(pred_test, 
#                                                 [x * alpha_m for x in pred_test_i])]
#         pred_train, pred_test = np.sign(pred_train), np.sign(pred_test)
        
#         # Return error rate in train and test set
#         #     return prf_ada, acc_ada, cm_ada
#         #     return tmp
#         # return get_error_rate(pred_train, Y_train), \
#         #         get_error_rate(pred_test, Y_test)
                
#     adaboost_clf(Y_train, X_train, Y_test, X_test, M, clf)
#     print ('main.prf', prf_ada)
#     print ('main.acc', acc_ada)
#     print ('main.cm', cm_ada)
#     return prf_ada, acc_ada, cm_ada
""" ADABOOST IMPLEMENTATION ================================================="""
