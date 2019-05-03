import numpy as np

def problem1 (A, B):
    return A + B

def problem2 (A, B, C):
    return A.dot(B) - C

def problem3 (A, B, C):
    return A * B + C.T

def problem4 (x, y):
    return np.inner(x,y)

def problem5 (A):
    return np.zeros_like(A)

def problem6 (A):
    return np.ones(A.shape[0])

def problem7 (A, alpha):
    return A + alpha * np.eye(A.shape[0])

def problem8 (A, i, j):
    return A[i,j]

def problem9 (A, i):
    return np.sum(A[i,:])

def problem10 (A, c, d):
    return np.mean(A[(A >= c) * (A <= d)])

def problem11 (A, k):
    values, vectors = np.linalg.eig(A)
    values = np.abs(values)
    indices = values.argsort()[::-1][:k]
    return vectors[:, indices]

def problem12 (A, x):
    return np.linalg.solve(A,x)

def problem13 (A, x):
    # x * A^-1  = ((A^-1)' * x')' 
    # x * A^-1  = ((A')^-1 * x')' 
    return np.linalg.solve(A.T, x.T).T
