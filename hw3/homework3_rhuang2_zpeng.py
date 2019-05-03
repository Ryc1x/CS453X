import numpy as np
import math
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import transform

'''
W:      785*10 weight matrix
X:      784*N or 28*28*N images matrix
Xtilde: 785*N images matrix
Y:      N*10 ground-truth matrix
Yhat:   N*10 prediction matrix
'''

# PART ONE: SOFTMAX REGRESSION #
def append1s(X):
    _, N = X.shape
    return np.vstack((X, np.ones(N)))

def fPC(W, Xtilde, Y):
    '''Return the percentage correctness of weights'''
    Yhat = softmax(W, Xtilde) #Xtilde.T.dot(W) # N*10 matrix
    predictIndices = np.argmax(Yhat, axis=1)
    actualIndices = np.argmax(Y, axis=1)
    return np.sum(1*(predictIndices == actualIndices)) / len(Y)

def fCE(W, Xtilde, Y):
    '''Return cross-entropy loss function of given weight'''
    Yhat = softmax(W, Xtilde) # N*10 matrix
    sum = 0
    for i in range(10):
        sum += Y[:,i].dot(np.log(Yhat[:,i]))
    return -sum / len(Y)

def gradfCE(W, Xtilde, Y):
    '''Return the gradient of fCE of given weight'''
    Yhat = softmax(W, Xtilde) # N*10 matrix
    return Xtilde.dot(Yhat-Y) / len(Y)

def softmax(W, Xtilde):
    '''Return Yhat(N*10) based on images and weights'''
    Z = W.T.dot(Xtilde) # 10*N matrix
    Zexp = np.exp(Z)
    sumZexp = np.sum(Zexp, axis=0) # 1*N vector
    return (Zexp / sumZexp[None,:]).T

def SGD(Xtilde, Y):
    '''Perform Stochastic gradient descent to minimize fCE'''
    n = len(Y)
    ntilde = 100
    EPOCH = 100
    ROUNDS = math.ceil(n/ntilde)
    epsilon = 1

    W = 0.001 * np.random.randn(len(Xtilde), 10)
    for e in range(EPOCH):
        shuffledRounds = np.arange(ROUNDS)
        np.random.shuffle(shuffledRounds)
        for r in shuffledRounds:
            if r % 10 == 0: # every 10 batches, scale epsilon by 0.98
                epsilon *= 0.98
            start = 100*r
            end = 100*(r+1)
            gradient = gradfCE(W, Xtilde[:, start:end], Y[start:end, :])
            W = W - epsilon * gradient
        if EPOCH - e <= 20:
            print("Iteration:", e, " | fCE:", fCE(W, Xtilde, Y), " | fPC:", fPC(W, Xtilde, Y))
    return W    

def visualize(vector):
    '''Visualize the given vector (image/weight)'''
    im = vector.reshape(28,28)
    _, ax = plt.subplots(1)
    ax.imshow(im, cmap='gray')
    plt.show()

# PART TWO: DATA AUGMENTATION #
def enlarge(X):
    n, m, N = X.shape
    largeX = np.zeros((3*n, 3*m, N), X.dtype)
    largeX[n:2*n, m:2*m, :] = X
    return largeX

def translate(X, dx=0, dy=0):
    n, m, _ = X.shape
    largeX = enlarge(X)
    y = n-dy
    x = m-dx
    return largeX[y:y+n, x:x+m, :]

def rotate(X, degree):
    _, _, N = X.shape
    rotatedX = np.zeros_like(X)
    for i in range(N):
        rotatedX[:,:,i] = transform.rotate(X[:,:,i], degree)
    return rotatedX

def scaleDown(X):
    _, _, N = X.shape
    scaledX = np.zeros_like(X)
    for i in range(N):
        scaledX[1:27,1:27,i] = transform.rescale(X[:,:,i], 0.92, multichannel=False, mode='reflect', anti_aliasing=True)
    return scaledX

def scaleUp(X):
    _, _, N = X.shape
    scaledX = np.zeros_like(X)
    for i in range(N):
        scaledX[:,:,i] = transform.rescale(X[:,:,i], 1.08, multichannel=False, mode='reflect', anti_aliasing=True)[1:29, 1:29]
    return scaledX

def noise(X):
    n, m, N = X.shape
    return X + 0.0001 * np.random.randn(n, m, N)

def gauss_filter(X, sigma=1):
    _, _, N = X.shape
    filteredX = np.zeros_like(X)
    for i in range(N):
        filteredX[:,:,i] = ndimage.gaussian_filter(X[:,:,i], sigma)
    return filteredX


if __name__ == "__main__":
    # Load data
    Xtr = np.load("small_mnist_train_images.npy").T # 784*N
    Ytr = np.load("small_mnist_train_labels.npy") # N*10
    Xte = np.load("small_mnist_test_images.npy").T # 784*N
    Yte = np.load("small_mnist_test_labels.npy") # N*10
    Xtilde_tr = append1s(Xtr) # 785*N
    Xtilde_te = append1s(Xte) # 785*N

    W = SGD(Xtilde_tr, Ytr)

    print("Training Accuracy:", fPC(W, Xtilde_tr, Ytr))
    print("Testing Accuracy:", fPC(W, Xtilde_te, Yte))
    print("Training fCE:", fCE(W, Xtilde_tr, Ytr))
    print("Testing fCE:", fCE(W, Xtilde_te, Yte))

    augment = False
    if augment:
        Xtr_reshaped = Xtr.reshape(28,28,5000)
        # translate up/down by one pixel
        Xtr_translate_up = translate(Xtr_reshaped, dx=0, dy=-1)
        Xtr_translate_down = translate(Xtr_reshaped, dx=0, dy=1)
        # rotate CW/CCW by 5 degree
        Xtr_rotate_cw = rotate(Xtr_reshaped, -5)
        Xtr_rotate_ccw = rotate(Xtr_reshaped, 5)
        # scale up/down (1.15/0.85)
        Xtr_scaled_up = scaleUp(Xtr_reshaped)
        Xtr_scaled_down = scaleDown(Xtr_reshaped)
        # randomly add noise to the images
        Xtr_noise = noise(Xtr_reshaped)

        # NOTE: [not used] apply gaussian filter with sigma = 0.5/1
        # Xtr_gauss1 = gauss_filter(Xtr_reshaped, 0.5)
        # Xtr_gauss2 = gauss_filter(Xtr_reshaped, 1)

        # stack the matrices together
        Xtr_augmented = np.dstack((Xtr_translate_up, Xtr_translate_down, Xtr_rotate_cw, Xtr_rotate_ccw, \
                                   Xtr_noise, Xtr_scaled_up, Xtr_scaled_down)).reshape(784, 7*5000)
        Xtilde_tr_augmented = append1s(Xtr_augmented)
        Ytr_augmented = np.tile(Ytr, (7,1))

        W_aug = SGD(Xtilde_tr_augmented, Ytr_augmented)

        print("Augmented Training Accuracy:", fPC(W_aug, Xtilde_tr_augmented, Ytr_augmented))
        print("Augmented Testing Accuracy:", fPC(W_aug, Xtilde_te, Yte))
        print("Augmented Training fCE:", fCE(W_aug, Xtilde_tr_augmented, Ytr_augmented))
        print("Augmented Testing fCE:", fCE(W_aug, Xtilde_te, Yte))

