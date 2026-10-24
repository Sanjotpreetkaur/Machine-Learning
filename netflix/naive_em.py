from typing import Tuple
import numpy as np
from common import GaussianMixture

def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    n,d = X.shape
    K, _ = mixture.mu.shape
    post = np.zeros((n,K))
    ll=0
    for i in range(n):
        for j in range(K):
            likelihood = np.exp((-d*0.5*np.log(2*mixture.var[j]*np.pi) -(0.5/mixture.var[j])*(np.sum((X[i]-mixture.mu[j])**2))))
            post[i,j]= mixture.p[j]*likelihood
        total = (post[i,:]).sum()
        
        post[i,:] = post[i,:]/total
        ll = ll + np.log(total)
    print(post[K], total)    
    return post, ll

def mstep(X: np.ndarray, post: np.ndarray) -> GaussianMixture:
    n, d = X.shape
    n, K = post.shape
    n_hat = np.sum(post, axis=0)
    mu = np.zeros((K,d))
    var = np.zeros(K)
    p = n_hat/n
    for i in range(K):
        mu[i,:] = (np.sum(X*post[:,i]))/n_hat[i]
        var[i] = (np.sum((mu[i]-X)**2, axis=1)*post[:,i])
    return GaussianMixture(mu, var, p)
