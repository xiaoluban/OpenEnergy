"""
This is an alternative, more computationally efficient solution that
avoids two separate function calls to f(b) and hess(b).
Also see: https://github.com/scipy/scipy/issues/9265
"""

import numpy as np
from scipy.optimize import minimize

# ----------------------------------------- #
# --- Wrapper Class for Caching Hessian --- #
# ----------------------------------------- #

class ObjectiveWrapper:
    """
    Caches the gradient and hessian of the objective function.
    """
    def __init__(self, func):
        self.func = func
        self._grad = None
        self._hess = None
        self.x = None

    def __call__(self, x, *args):
        self.x = np.asarray(x).copy()
        fgh = self.func(x, *args)
        self._grad = fgh[1]
        self._hess = fgh[2]
        return fgh[0]

    def grad(self, x, *args):
        if self._grad is not None and np.alltrue(x == self.x):
            return self._grad
        else:
            self(x, *args)
            return self._grad

    def hess(self, x, *args):
        if self._hess is not None and np.alltrue(x == self.x):
            return self._hess
        else:
            self(x, *args)
            return self._hess
    def predict(self, x_predict, b_predict):
        # print('b', b)
        onehot_time = np.eye(x_predict.shape[1])
        bb = np.matmul(np.expand_dims(b_predict, 0), onehot_time)
        # bb = np.expand_dims(b_predict, 0)
        bbb = []
        for _ in range(x_predict.shape[0]):
            bbb.append(bb)
        bbb = np.vstack(bbb)
        X = x_predict
        Xb = X * bbb
        exp_Xb = np.exp(Xb)

        return exp_Xb
    # def predict(self, x_predict, b_predict):
    #     # Xb = np.dot(x_predict, b_predict)
    #     Xb = np.sum(x_predict * b_predict, 1)
    #     pred_y = np.exp(Xb)
    #     return pred_y
# ----------------------------- #
# --- Create Synthetic Data --- #
# ----------------------------- #

n = 1000   # number of datapoints
p = 5     # number of features

X = .3*np.random.randn(n, p)
true_b = np.random.randn(p)
y = np.random.poisson(np.exp(np.dot(X, true_b)))

# ----------------------------------------------- #
# --- Define Loss Function, Gradient, Hessian --- #
# ----------------------------------------------- #

def f(b):
    Xb = np.dot(X, b)
    exp_Xb = np.exp(Xb)
    loss = exp_Xb.sum() - np.dot(y, Xb)
    grad = np.dot(X.T, exp_Xb - y)
    t = exp_Xb[:, None] * X
    hess = np.dot(X.T, exp_Xb[:, None] * X)
    return loss, grad, hess

obj = ObjectiveWrapper(f)  # wrap objective function

# ----------------------------------------------- #
# --- Define Loss Function, Gradient, Hessian --- #
# ----------------------------------------------- #

x0 = np.zeros(p)
result = minimize(obj, x0, jac=obj.grad, hess=obj.hess, method='newton-cg')
# print('Estimated regression coeffs: {}'.format(result.x))
# print('True regression coeffs: {}'.format(true_b))