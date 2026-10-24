import sys
sys.path.append("..")
import utils
from utils import *
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sparse

def augment_feature_vector(X):
    column_of_ones = np.zeros([len(X), 1]) + 1
    return np.hstack((column_of_ones, X))

def compute_probabilities(X, theta, temp_parameter):
    z = np.dot(theta,X.transpose()/temp_parameter)
    c = np.max(z, axis=0)
    exp_z = np.exp(z-c)
    z_sum= np.sum(exp_z, axis=0)
    H = (1/z_sum)*exp_z
    return H   
def compute_cost_function(X, Y, theta, lambda_factor, temp_parameter):

    n, d = X.shape
    reg_term = (lambda_factor/2)*np.sum(theta*theta)
    prob=compute_probabilities(X, theta, temp_parameter)
    loss_prob = np.choose(Y, prob)
    loss = reg_term - (1/n)*np.sum(np.log(loss_prob))
    return loss


def run_gradient_descent_iteration(X, Y, theta, alpha, lambda_factor, temp_parameter):

    n, d = X.shape
    k = np.size(theta,0)
    probabilities = compute_probabilities(X, theta, temp_parameter)
    
    sparse_Y = sparse.coo_matrix(([1]*n, (Y, range(n))), shape=(k,n)).toarray()
    constant = 1/(temp_parameter*n)
    reg_term = lambda_factor*theta
    grad_des = reg_term - constant*((np.dot((sparse_Y-probabilities), X)))

    theta = theta - alpha*grad_des
    return theta


def update_y(train_y, test_y):

    train_y_mod3 = np.mod(train_y, 3)
    test_y_mod3 = np.mod(test_y, 3)
    return train_y_mod3, test_y_mod3

def compute_test_error_mod3(X, Y, theta, temp_parameter):

    argmax_h = get_classification(X, theta, temp_parameter)
    return 1 - np.mean(np.mod(argmax_h,3) == Y)


def softmax_regression(X, Y, temp_parameter, alpha, lambda_factor, k, num_iterations):

    X = augment_feature_vector(X)
    theta = np.zeros([k, X.shape[1]])
    cost_function_progression = []
    for i in range(num_iterations):
        cost_function_progression.append(compute_cost_function(X, Y, theta, lambda_factor, temp_parameter))
        theta = run_gradient_descent_iteration(X, Y, theta, alpha, lambda_factor, temp_parameter)
    return theta, cost_function_progression

def get_classification(X, theta, temp_parameter):

    X = augment_feature_vector(X)
    probabilities = compute_probabilities(X, theta, temp_parameter)
    return np.argmax(probabilities, axis = 0)

def plot_cost_function_over_time(cost_function_history):
    plt.plot(range(len(cost_function_history)), cost_function_history)
    plt.ylabel('Cost Function')
    plt.xlabel('Iteration number')
    plt.show()

def compute_test_error(X, Y, theta, temp_parameter):
    error_count = 0.
    assigned_labels = get_classification(X, theta, temp_parameter)
    return 1 - np.mean(assigned_labels == Y)
