import numpy as np
def polynomial_kernel(X, Y, c, p):
    kernel = (np.dot(X, Y.transpose()) + c)
    return np.power(kernel,p)
    raise NotImplementedError

def rbf_kernel(X, Y, gamma):
    return np.exp(-gamma*np.power((np.sum(X[:, np.newaxis]-Y, axis = -1)), 2))
    raise NotImplementedError
