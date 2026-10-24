import numpy as np

def closed_form(X, Y, lambda_factor):

    identity_matrix = np.identity(np.size(X, 1))
    left_term = np.add(np.dot(X.transpose(), X), lambda_factor*identity_matrix)
    theta = np.dot(np.dot(Y, X), np.linalg.inv(left_term))
    return theta


