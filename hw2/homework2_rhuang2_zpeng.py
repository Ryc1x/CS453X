import numpy as np
import matplotlib.patches as patches
import matplotlib.pyplot as plt

# Given an array of faces (N x M x M, where N is number of examples and M is number of pixes along each axis),
# return a design matrix Xtilde ((M**2 + 1) x N) whose last row contains all 1s.
def reshapeAndAppend1s (faces):
    N, M, M = faces.shape
    return np.vstack((faces.reshape(N, M**2).T, np.ones(N)))

# Given a vector of weights w, a design matrix Xtilde, and a vector of labels y, return the (unregularized)
# MSE.
def fMSE (w, Xtilde, y):
    yhat = Xtilde.T.dot(w)
    return np.sum((y-yhat)**2) / (2*len(y))

# Given a vector of weights w, a design matrix Xtilde, and a vector of labels y, and a regularization strength
# alpha (default value of 0), return the gradient of the (regularized) MSE loss.
def gradfMSE (w, Xtilde, y, alpha = 0.):
    n = len(y)
    w_no_bias = np.copy(w)
    w_no_bias[-1] = 0
    return (Xtilde.dot(Xtilde.T.dot(w)-y) + (alpha*w_no_bias)) / n

# Given a design matrix Xtilde and labels y, train a linear regressor for Xtilde and y using the analytical solution.
def method1 (Xtilde, y):
    return np.linalg.solve(Xtilde.dot(Xtilde.T), Xtilde.dot(y))

# Given a design matrix Xtilde and labels y, train a linear regressor for Xtilde and y using gradient descent on fMSE.
def method2 (Xtilde, y):
    return gradientDescent(Xtilde, y)

# Given a design matrix Xtilde and labels y, train a linear regressor for Xtilde and y using gradient descent on fMSE
# with regularization.
def method3 (Xtilde, y):
    ALPHA = 0.1
    return gradientDescent(Xtilde, y, ALPHA)

# Helper method for method2 and method3.
def gradientDescent (Xtilde, y, alpha = 0.):
    EPSILON = 3e-3  # Step size aka learning rate
    T = 5000  # Number of gradient descent iterations

    w = 0.01 * np.random.randn(len(Xtilde))
    for i in range(T):
        if i%1000 == 0: print(i, "iterations")
        gradient = gradfMSE(w, Xtilde, y, alpha)
        w = w - EPSILON * gradient
    return w

# Visualize the weights (or a single image)
def visualize(w):
    im = w[:-1].reshape(48,48)
    fig,ax = plt.subplots(1)
    ax.imshow(im, cmap='gray')
    plt.show()


if __name__ == "__main__":
    # Load data
    Xtilde_tr = reshapeAndAppend1s(np.load("age_regression_Xtr.npy"))
    ytr = np.load("age_regression_ytr.npy")
    Xtilde_te = reshapeAndAppend1s(np.load("age_regression_Xte.npy"))
    yte = np.load("age_regression_yte.npy")

    w1 = method1(Xtilde_tr, ytr)
    w2 = method2(Xtilde_tr, ytr)
    w3 = method3(Xtilde_tr, ytr)

    # Report fMSE cost using each of the three learned weight vectors
    print("Method\tfMSE-training\tfMSE-testing")
    print("Analy.", fMSE(w1, Xtilde_tr, ytr), fMSE(w1, Xtilde_te, yte), sep="\t")
    print("GD", fMSE(w2, Xtilde_tr, ytr), fMSE(w2, Xtilde_te, yte), sep="\t")
    print("GD reg.", fMSE(w3, Xtilde_tr, ytr), fMSE(w3, Xtilde_te, yte), sep="\t")
    
    # Visualizations
    # for part(c), show top 5 most egregious errors (change True/False to toggle output)
    if False:
        top5_idx = np.argsort(abs(yte - Xtilde_te.T.dot(w3)))[::-1][:5]
        for i in top5_idx:
            visualize(Xtilde_te[:,i])