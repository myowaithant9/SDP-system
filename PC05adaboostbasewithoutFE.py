import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
# from sklearn.cross_validation import train_test_split

from sklearn.model_selection import cross_validate                                          
from sklearn.model_selection import cross_val_predict                          
from sklearn.model_selection import cross_val_score 
from sklearn.model_selection import train_test_split

import openpyxl


from sklearn.datasets import make_hastie_10_2
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix

""" HELPER FUNCTION: GET ERROR RATE ========================================="""
def get_error_rate(pred, Y):
    return sum(pred != Y) / float(len(Y))

""" HELPER FUNCTION: PRINT ERROR RATE ======================================="""
def print_error_rate(err):
    print ('Error rate: Training: %.4f - Test: %.4f' % err)

""" HELPER FUNCTION: GENERIC CLASSIFIER ====================================="""
def generic_clf(Y_train, X_train, Y_test, X_test, clf):
    clf.fit(X_train,Y_train)
    pred_train = clf.predict(X_train)
    pred_test = clf.predict(X_test)
    print('pred_train for generic', pred_train)
    print('pred_test for generic', pred_test)

    print('PC05precision_recall_fscore_support avg none: {}', precision_recall_fscore_support(Y_test, pred_test, average=None))
    print('Accuracy for Generic PC05', accuracy_score(Y_test, pred_test))
    
    cm = confusion_matrix(Y_test, pred_test)
    print("\nConfusion Matrix PC05", cm)

    return get_error_rate(pred_train, Y_train), \
           get_error_rate(pred_test, Y_test)
    
""" ADABOOST IMPLEMENTATION ================================================="""
def adaboost_clf(Y_train, X_train, Y_test, X_test, M, clf):
    n_train, n_test = len(X_train), len(X_test)
    # print('ntrain', n_train)
    # print('ntest', n_test)
    # Initialize weights
    w = np.ones(n_train) / n_train
    # print('w first one', w)
    pred_train, pred_test = [np.zeros(n_train), np.zeros(n_test)]
    # print('pred_train', pred_train)
    # print('pred_test', pred_test)

    for i in range(M):
        # Fit a classifier with the specific weights
        clf.fit(X_train, Y_train, sample_weight = w)
        pred_train_i = clf.predict(X_train)
        pred_test_i = clf.predict(X_test)

        print(str(i) + 'PC05 precision_recall_fscore_support avg none: {}', precision_recall_fscore_support(Y_test, pred_test_i, average=None))

        print('PC05 Accuracy for'+ str(i) +'-->', accuracy_score(Y_test, pred_test_i))
        
        cm = confusion_matrix(Y_test, pred_test_i)
        print("\nConfusion Matrix PC05", cm)

        # print('pred_train_'+ str(i) +'-->', pred_train_i)
        # print('pred_test_i'+ str(i) +'-->', pred_test_i)
        # Indicator function
        miss = [int(x) for x in (pred_train_i != Y_train)]
        # print('miss'+ str(i) +'-->', miss)
        # Equivalent with 1/-1 to update weights
        miss2 = [x if x==1 else -1 for x in miss]
        # print('miss2', miss2)

        # Error
        err_m = np.dot(w,miss) / sum(w)
        # print('err_m'+ str(i) +'-->', err_m)

        # Alpha
        alpha_m = 0.5 * np.log( (1 - err_m) / float(err_m))
        # print('alpha_m'+ str(i) +'-->', alpha_m)

        # New weights
        w = np.multiply(w, np.exp([float(x) * alpha_m for x in miss2]))
        # print('w'+ str(i) +'-->', w)
        # print('a')
        # a = zip(pred_train,(x * alpha_m for x in pred_train_i))
        # a = zip(pred_train,(x * alpha_m for x in pred_train_i))

        # print('a' , a)
        # Add to prediction
        pred_train = [sum(x) for x in zip(pred_train, 
                                          [x * alpha_m for x in pred_train_i])]
        pred_test = [sum(x) for x in zip(pred_test, 
                                         [x * alpha_m for x in pred_test_i])]
        # print('pred_train calculate'+ str(i) +'-->', np.sign(pred_train))
        # print('pred_test'+ str(i) +'-->', pred_test)
        # pred_train_t, pred_test_t = np.sign(pred_train), np.sign(pred_test)
        # print('error_rate_train'+str(i)+'-->', get_error_rate(pred_train_t, Y_train))
        # print('error_rate_test'+str(i)+'-->', get_error_rate(pred_test_t, Y_test))
    
    pred_train, pred_test = np.sign(pred_train), np.sign(pred_test)
    # Return error rate in train and test set
    return get_error_rate(pred_train, Y_train), \
           get_error_rate(pred_test, Y_test)

""" PLOT FUNCTION ==========================================================="""
def plot_error_rate(er_train, er_test):
    df_error = pd.DataFrame([er_train, er_test]).T
    df_error.columns = ['Training', 'Test']
    plot1 = df_error.plot(linewidth = 3, figsize = (8,6),
            color = ['lightblue', 'darkblue'], grid = True)
    plot1.set_xlabel('Number of iterations', fontsize = 12)
    plot1.set_xticklabels(range(0,450,50))
    plot1.set_ylabel('Error rate', fontsize = 12)
    plot1.set_title('Error rate vs number of iterations', fontsize = 16)
    plt.axhline(y=er_test[0], linewidth=1, color = 'red', ls = 'dashed')

""" Preprocess SCRIPT ============================================================="""
def preprocess(dataset_path):
    # dataset_path = 'PC3.csv'
    dataset = pd.read_csv(dataset_path)
    np.set_printoptions(formatter={'float_kind':'{:0f}'.format})

    print('Dataset')
    print(dataset)

    # indexNames = dataset[ dataset['DECISION_DENSITY'] == '?' ].index 
    # dataset.drop(indexNames , inplace=True)

    col_count = 38
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
    return feature_data, target_data

""" Discretization SCRIPT ============================================================="""
def discretize(feature_data):
    from sklearn.preprocessing import KBinsDiscretizer

    est = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='kmeans')  #uniform, quantile, kmeans
    est.fit(feature_data)
    discretize_data = est.transform(feature_data)
    discretize_data = discretize_data.astype(int)
    # print(discretize_data.shape)
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
    # concat_data = np.arange(12375).reshape(1125,11)
    concat_data = np.concatenate((selected_data, target_data[:,None]), axis=1)
    return concat_data

""" MAIN SCRIPT ============================================================="""
if __name__ == '__main__':
    dataset_path = 'MDP csv/PC5clean.csv'

    # dataset_path = 'MDP csv/PC05.csv'
    feature_data, target_data = preprocess(dataset_path)
    print('feature_data')
    print(feature_data)

    print('target_data')
    print(target_data)

    discretize_data = discretize(feature_data)
    print('\n')
    print ("*** Discretize Data ***")
    print (discretize_data)
    print(discretize_data.shape)

    # num_fea = 10
    # feature_extraction = feature_extract(discretize_data, target_data, num_fea)
    # print('\n')
    # print("*** Selected Feature ***")
    # print(feature_extraction)
    
    # selected_data = discretize_data[:, [0, 16, 9,  8, 17, 35,  2, 22, 20, 19]]  #=> transform manual to auto
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

    X_train = train_data[:,0:38]
    Y_train = train_data[:,38].astype('int')

    X_test = test_data[:,0:38]
    Y_test = test_data[:,38].astype('int')
    
    # Read data
    # x, y = make_hastie_10_2()
    # df = pd.DataFrame(x)
    # df['Y'] = y

    # Split into training and test set
    # train, test = train_test_split(df, test_size = 0.2)
    # print('train', train)
    # print('test', test)

    # X_train, Y_train = train.iloc[:,:-1], train.iloc[:,-1]
    # X_test, Y_test = test.iloc[:,:-1], test.iloc[:,-1]

    
    
    from sklearn.svm import SVC
    # C = 1.0

    # clf_tree = svm.SVC(kernel='rbf', gamma=0.7, C=C)
    # clf_tree = SVC(kernel='rbf', gamma='scale', random_state = 1)  

    clf_tree = SVC(kernel='rbf', random_state=0, gamma=.01, C=10000)
    # Fit a simple decision tree first
    # clf_tree = DecisionTreeClassifier(max_depth = 1, random_state = 1)
    er_tree = generic_clf(Y_train, X_train, Y_test, X_test, clf_tree)
    print('clf_tree', clf_tree)
    print('er_tree', er_tree)
    
    # Fit Adaboost classifier using a decision tree as base estimator
    # Test with different number of iterations
    er_train, er_test = [er_tree[0]], [er_tree[1]]
    print('er_train', er_train)
    print('er_test', er_test)

    x_range = range(0, 40, 1)
    for i in x_range:    
        er_i = adaboost_clf(Y_train, X_train, Y_test, X_test, i, clf_tree)
        print('er_i'+ str(i) +'-->', er_i)

        er_train.append(er_i[0])
        er_test.append(er_i[1])
    
    # Compare error rate vs number of iterations
    plot_error_rate(er_train, er_test)