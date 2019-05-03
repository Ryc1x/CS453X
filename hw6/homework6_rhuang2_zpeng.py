import numpy as np
import math
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import scipy.optimize
from sklearn import decomposition

NUM_INPUT = 784  # Number of input neurons
NUM_OUTPUT = 10  # Number of output neurons
NUM_CHECK = 5  # Number of examples on which to check the gradient

HYPER_NEURON = 50
HYPER_EPSILON = 0.01
HYPER_BATCH = 32
HYPER_EPOCH = 200


# Given a vector w containing all the weights and biased vectors, extract
# and return the individual weights and biases W1, b1, W2, b2.
# This is useful for performing a gradient check with check_grad.
def unpack (w):
    idx1 = NUM_INPUT*HYPER_NEURON            # W1
    idx2 = idx1 + HYPER_NEURON               # b1
    idx3 = idx2 + HYPER_NEURON*NUM_OUTPUT    # W2
    W1 = w[    : idx1].reshape(HYPER_NEURON, NUM_INPUT)
    b1 = w[idx1: idx2].reshape(HYPER_NEURON)
    W2 = w[idx2: idx3].reshape(NUM_OUTPUT, HYPER_NEURON)
    b2 = w[idx3:     ].reshape(NUM_OUTPUT)
    return W1, b1, W2, b2

# Given individual weights and biases W1, b1, W2, b2, concatenate them and
# return a vector w containing all of them.
# This is useful for performing a gradient check with check_grad.
def pack (W1, b1, W2, b2):
    return np.concatenate((W1.flatten(), b1.flatten(), W2.flatten(), b2.flatten()))

# Given a Z, return the derivative of ReLU of Z
def relu_prime(Z):
    return 1 * (Z > 0)

# Give a Z, return the ReLu of Z
def relu (z):
    return z * (z > 0)

# Load the images and labels from a specified dataset (train or test).
def loadData (which):
    images = np.load("mnist_{}_images.npy".format(which))
    labels = np.load("mnist_{}_labels.npy".format(which))
    return images, labels

# Given training images X, associated labels Y, and a vector of combined weights
# and bias terms w, compute and return the cross-entropy (CE) loss. You might
# want to extend this function to return multiple arguments (in which case you
# will also need to modify slightly the gradient check code below).
def fCE (X, Y, w):
    W1, b1, W2, b2 = unpack(w)
    Z1 = W1.dot(X.T) + b1[: ,None]
    h1 = relu(Z1)
    Z2 = W2.dot(h1) + b2[: ,None]
    Yhat = softmax(Z2)
    return -np.sum(Y * np.log(Yhat)) / len(Y)

# Return the percentage correctness of weights
def fPC(X, Y, w):
    W1, b1, W2, b2 = unpack(w)
    Z1 = W1.dot(X.T) + b1[: ,None]
    h1 = relu(Z1)
    Z2 = W2.dot(h1) + b2[: ,None]
    Yhat = softmax(Z2)
    predictIndices = np.argmax(Yhat, axis=1)
    actualIndices = np.argmax(Y, axis=1)
    return np.sum(1*(predictIndices == actualIndices)) / len(Y)

# Give a Z, return the softmax of Z
def softmax(Z):
    Zexp = np.exp(Z)
    sumZexp = np.sum(Zexp, axis=0) # 1*N vector
    return (Zexp / sumZexp[None,:]).T

# Given training images X, associated labels Y, and a vector of combined weights
# and bias terms w, compute and return the gradient of fCE. You might
# want to extend this function to return multiple arguments (in which case you
# will also need to modify slightly the gradient check code below).
def gradCE (X, Y, w, alpha1=0, beta1=0, alpha2=0, beta2=0):
    W1, b1, W2, b2 = unpack(w)
    Z1 = W1.dot(X.T) + b1[: ,None]
    h1 = relu(Z1)
    Z2 = W2.dot(h1) + b2[: ,None]
    Yhat = softmax(Z2)

    grad_b2fCE = np.mean(Yhat-Y, axis=0)  # 10*1
    grad_W2fCE = ((Yhat-Y).T.dot(h1.T) + alpha2 * W2 + beta2 * np.sign(W2)) / len(Y)  # 10*40
    G_transpose = np.multiply((Yhat-Y).dot(W2), relu_prime(Z1.T)) # N*40
    grad_b1fCE = np.mean(G_transpose, axis=0) # 40*1
    grad_W1fCE = (G_transpose.T.dot(X) + alpha1 * W1 + beta1 * np.sign(W1)) / len(Y) # 40*784

    return pack(grad_W1fCE, grad_b1fCE, grad_W2fCE, grad_b2fCE)

# Given X, Y, and initialized w
# Perform Stochastic gradient descent to minimize fCE
# and return a sequence of w's
def SGD (X, Y, w, showAccuracy=False, testX=None, testY=None):
    n = len(Y)
    ROUNDS = math.ceil(n/HYPER_BATCH)
    W1, b1, W2, b2 = unpack(w)
    wList = []

    for e in range(HYPER_EPOCH):
        shuffledRounds = np.arange(ROUNDS)
        np.random.shuffle(shuffledRounds)
        for r in shuffledRounds:
            # if r%10 == 0: # every 10 batches, scale epsilon by 0.95
            #     epsilon *= 0.95
            start = HYPER_BATCH*r
            end = HYPER_BATCH*(r+1)
            gradientW = gradCE(X[start:end, :], Y[start:end, :], w, 0.001, 0.001, 0.001, 0.001)
            gradW1, gradb1, gradW2, gradb2 = unpack(gradientW)
            W1 = W1 - HYPER_EPSILON * gradW1
            b1 = b1 - HYPER_EPSILON * gradb1
            W2 = W2 - HYPER_EPSILON * gradW2
            b2 = b2 - HYPER_EPSILON * gradb2
            w = pack(W1, b1, W2, b2)
        wList.append(w)
        if showAccuracy and HYPER_EPOCH - e <= 20:
            print("Iteration:", e, " | fCE:", fCE(testX, testY, w), " | fPC:", fPC(testX, testY, w))
    return wList   

# Given training and testing datasets and an initial set of weights/biases b,
# train the NN. Then return the sequence of w's obtained during SGD.
def train (trainX, trainY, testX, testY, w):
    return SGD(trainX, trainY, w, showAccuracy=True, testX=testX, testY=testY) # 30, 0.01, 64, 200


# Give a validation data and label,
# find the best hyperparameters
def findBestHyperparameters(validX, validY):
    fCELowest = 1000
    validXCopy = validX
    validYCopy = validY
    neuron = [30, 40, 50]  # Number of hidden neurons {30; 40; 50}
    epsilon = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
    mini_batch = [16, 32, 64, 128, 256] # 16; 32; 64; 128; 256
    epoch = [100, 200, 500]
    # REGULAR 

    for neu in neuron[1: ]: # NEURON in gradCE
        for esp in epsilon[1: 3]:
            for bat in mini_batch[1: 4]:
                for epo in epoch[ :2]:
                    validX = validXCopy
                    validY = validYCopy
                    
                    global HYPER_NEURON, HYPER_EPSILON, HYPER_BATCH, HYPER_EPOCH
                    temp_neu = HYPER_NEURON
                    temp_eps = HYPER_EPSILON
                    temp_bat = HYPER_BATCH
                    temp_epo = HYPER_EPOCH
                    HYPER_NEURON = neu
                    HYPER_EPSILON = esp
                    HYPER_BATCH = bat
                    HYPER_EPOCH = epo

                    # Initialize weights randomly
                    W1 = 2*(np.random.random(size=(NUM_INPUT, neu))/NUM_INPUT**0.5) - 1./NUM_INPUT**0.5
                    b1 = 0.01 * np.ones(neu)
                    W2 = 2*(np.random.random(size=(neu, NUM_OUTPUT))/neu**0.5) - 1./neu**0.5
                    b2 = 0.01 * np.ones(NUM_OUTPUT)
                    w = pack(W1, b1, W2, b2)

                    wCurrent = SGD(validX, validY, w)[-1] # alpha1,2 beta1,2
                    fCECurrent = fCE(validX, validY, wCurrent)
                    fPCCurrent = fPC(validX, validY, wCurrent)
                    print(neu, esp, bat, epo, sep='\t')
                    if fCECurrent < fCELowest:
                        print("OPTIMIZED:", " | fCE:", fCECurrent, " | fPC:", fPCCurrent)
                        fCELowest = fCECurrent
                    else:
                        HYPER_NEURON = temp_neu
                        HYPER_EPSILON = temp_eps
                        HYPER_BATCH = temp_bat
                        HYPER_EPOCH = temp_epo

    print("BEST HYPER PARAMETERS: ")
    print(HYPER_NEURON, HYPER_EPSILON, HYPER_BATCH, HYPER_EPOCH, sep='\t')  # 40      0.5     128     200

# PART 2
def plotSGDPath (trainX, trainY, ws):
    # PCA
    pca = decomposition.PCA(n_components=2)
    pca.fit(ws)
    ws_reduced = ws[::]
    ws_transformed = pca.transform(ws_reduced)
    x_max, y_max = np.max(ws_transformed, axis=0) + 4
    x_min, y_min = np.min(ws_transformed, axis=0) - 4
    x_step = (x_max-x_min) / 10
    y_step = (y_max-y_min) / 10

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # Compute the CE loss on a grid of points (corresonding to different w).
    axis1 = np.arange(x_min, x_max, x_step)  # Just an example
    axis2 = np.arange(y_min, y_max, y_step)  # Just an example
    Xaxis, Yaxis = np.meshgrid(axis1, axis2)
    Zaxis = np.zeros((len(axis1), len(axis2)))
    # coords = np.transpose([np.tile(axis1, len(axis2)), np.repeat(axis2, len(axis1))])
    # weight_grids = pca.inverse_transform(coords)
    for i in range(len(axis2)):
        # print("i:", i)
        for j in range(len(axis1)):
            Zaxis[i,j] = fCE(trainX, trainY, pca.inverse_transform([Xaxis[i,j], Yaxis[i,j]]))
    ax.plot_surface(Xaxis, Yaxis, Zaxis, alpha=0.6)  # Keep alpha < 1 so we can see the scatter plot too.


    # Now superimpose a scatter plot showing the weights during SGD.
    Xaxis = ws_transformed[:,0]
    Yaxis = ws_transformed[:,1]
    Zaxis = np.zeros(len(Xaxis))
    # colors = np.arange(len(Xaxis))
    for i in range(len(Xaxis)):
        Zaxis[i] = fCE(trainX, trainY, ws_reduced[i])
    ax.scatter(Xaxis, Yaxis, Zaxis, color='r')

    ax.set_zlim3d(0, 0.5)
    plt.show()

if __name__ == "__main__":
    # Load data
    if "trainX" not in globals():
        trainX, trainY = loadData("train")
        testX, testY = loadData("test")
        validX, validY = loadData("validation")

    # Initialize weights randomly
    W1 = 2*(np.random.random(size=(NUM_INPUT, HYPER_NEURON))/NUM_INPUT**0.5) - 1./NUM_INPUT**0.5
    b1 = 0.01 * np.ones(HYPER_NEURON)
    W2 = 2*(np.random.random(size=(HYPER_NEURON, NUM_OUTPUT))/HYPER_NEURON**0.5) - 1./HYPER_NEURON**0.5
    b2 = 0.01 * np.ones(NUM_OUTPUT)
    wCheck = pack(W1, b1, W2, b2)

    # # Check that the gradient is correct on just a few examples (randomly drawn).
    print("_______________ Check Gradient _______________")
    idxs = np.random.permutation(trainX.shape[0])[0:NUM_CHECK]
    print(scipy.optimize.check_grad(lambda w_: fCE(np.atleast_2d(trainX[idxs,:]), np.atleast_2d(trainY[idxs,:]), w_), \
                                    lambda w_: gradCE(np.atleast_2d(trainX[idxs,:]), np.atleast_2d(trainY[idxs,:]), w_), \
                                    wCheck))

    # Find best hyperparameters
    print("_________ Find best hyperparameters __________")
    print("Neuron\tEpsilon\tBatch\tEpoch")
    findBestHyperparameters(validX, validY) 

    W1 = 2*(np.random.random(size=(NUM_INPUT, HYPER_NEURON))/NUM_INPUT**0.5) - 1./NUM_INPUT**0.5
    b1 = 0.01 * np.ones(HYPER_NEURON)
    W2 = 2*(np.random.random(size=(HYPER_NEURON, NUM_OUTPUT))/HYPER_NEURON**0.5) - 1./HYPER_NEURON**0.5
    b2 = 0.01 * np.ones(NUM_OUTPUT)
    wTrain = pack(W1, b1, W2, b2)
    # Train the network and obtain the sequence of w's obtained using SGD.
    print("______________ Starts Training _______________")
    ws = train(trainX, trainY, testX, testY, wTrain)
    w = ws[-1]
    
    print("_____________ Training Results _______________")
    print("Training Accuracy:", fPC(trainX, trainY, w))
    print("Testing Accuracy:", fPC(testX, testY, w))
    print("Training fCE:", fCE(trainX, trainY, w))
    print("Testing fCE:", fCE(testX, testY, w))

    # Plot the SGD trajectory
    print("_____________ Plotting SGD Path ______________")
    plotSGDPath(trainX, trainY, ws)
