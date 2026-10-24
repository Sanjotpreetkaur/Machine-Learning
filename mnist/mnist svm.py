import numpy as np
from sklearn.svm import LinearSVC
from sklearn.svm import SVC

def one_vs_rest_svm(train_x, train_y, test_x):
    for i in train_y:
        if i>0:
            i=1
    clsf = LinearSVC(random_state = 0, C=0.1)
    clsf.fit(train_x, train_y)
    pred_test_y = clsf.predict(test_x)
    return pred_test_y

def multi_class_svm(train_x, train_y, test_x):

    clsf = LinearSVC(random_state = 0, C=0.1)
    clsf.fit(train_x, train_y)
    pred_test_y = clsf.predict(test_x)
    return pred_test_y
    
def multi_class_svm_10pca(train_x, train_y, test_x, random_state, kernel, degree):
    
    clsf = SVC(random_state = random_state, kernel = kernel, degree= degree)
    clsf.fit(train_x, train_y)
    pred_test_y = clsf.predict(test_x)
    return pred_test_y

def compute_test_error_svm(test_y, pred_test_y):
    return 1 - np.mean(pred_test_y == test_y)

