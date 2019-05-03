import numpy as np
import matplotlib.pyplot as plt

def kLargestEig (A, k):
    values, vectors = np.linalg.eig(A)
    # Eigenvalues of every PSD matrix A are always non-negative
    indices = values.argsort()[::-1][:k] 
    return vectors[:, indices]

def PCA(Xtilde, Y):
    # Shift X such that var(X) stays the same but mean(X) = 0
    Xtilde = Xtilde - np.mean(Xtilde, axis=1)[:, None]
    # Find out the largest 2 eigen vector
    vectors = kLargestEig(Xtilde.dot(Xtilde.T), 2).T
    # Setup1: scatter plot
    x = vectors[0].dot(Xtilde)
    y = vectors[1].dot(Xtilde)
    
    # Setup2: takes the opposite value of the vectors, which will produce the same graph as provide in the homework instruction
    # x = -vectors[0].dot(Xtilde)
    # y = -vectors[1].dot(Xtilde)
    colors = np.argmax(Y, axis=1)**0.5
    plt.scatter(x, y, s=0.5, c=colors, alpha=0.8)
    plt.show()

if __name__ == "__main__":
    # Load data
    X = np.load("small_mnist_test_images.npy").T # 784*N
    Y = np.load("small_mnist_test_labels.npy") # N*10
    PCA(X, Y)