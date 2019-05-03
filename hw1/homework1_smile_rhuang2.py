import numpy as np
import matplotlib.patches as patches
import matplotlib.pyplot as plt

def fPC (y, yhat):
    return np.sum(1 * (y == yhat)) / len(y)

def measureAccuracyOfPredictors (predictors, X, y):
    sum_g = np.zeros(len(X))
    for p in predictors:
        pix1 = X[:, p[0], p[1]]
        pix2 = X[:, p[2], p[3]]
        features = pix1 - pix2
        g = 1 * (features > 0)  # convert boolean to int
        sum_g += g
    yhat = 1 * (sum_g/len(predictors) > 0.5)
    return fPC(y, yhat)


def stepwiseRegression (trainingFaces, trainingLabels, testingFaces, testingLabels, n=None):
    predictors = []  # each predictor is a touple: (r1, c1, r2, c2)
    if n is not None: # take first n testing faces
        trainingFaces = trainingFaces[:n,:,:]
        trainingLabels = trainingLabels[:n]

    imgrange = range(24)
    for j in range(5):
        # find next best predctor
        sum_g = np.zeros(len(trainingFaces))

        # sum features from previous predictors
        for p in predictors:
            pix1 = trainingFaces[:, p[0], p[1]]
            pix2 = trainingFaces[:, p[2], p[3]]
            features = pix1 - pix2
            g = 1 * (features > 0)  # convert boolean to int
            sum_g += g

        # find the next best predictor
        best_corr = 0
        best_pred = (0,0,0,0)
        for r1 in imgrange:
            for c1 in imgrange:
                for r2 in imgrange:
                    for c2 in imgrange:
                        # new_predictors = predictors + [(r1, c1, r2, c2)]
                        # corr = measureAccuracyOfPredictors(new_predictors, trainingFaces, trainingLabels)
                        pix1 = trainingFaces[:,r1,c1]
                        pix2 = trainingFaces[:,r2,c2]
                        new_features = pix1 - pix2
                        g = 1 * (new_features > 0)  # convert boolean to int
                        yhat = 1 * ((sum_g + g)/(j+1) > 0.5)
                        corr = fPC(trainingLabels, yhat)
                        if corr > best_corr:
                            best_corr = corr 
                            best_pred = (r1,c1,r2,c2)
                        
        # add into predictors
        predictors.append(best_pred)
        # print(predictors)
        # print("Testing accuracy: ", measureAccuracyOfPredictors(predictors, testingFaces, testingLabels))

    show = True
    if show:
        # Show an arbitrary test image in grayscale
        im = testingFaces[0,:,:]
        fig,ax = plt.subplots(1)
        ax.imshow(im, cmap='gray')
        for idx, p in enumerate(predictors):
            r1, c1, r2, c2 = p
            # Show r1,c1
            rect = patches.Rectangle((c1-0.5,r1-0.5),1,1,linewidth=2,edgecolor='r',facecolor='none')
            ax.add_patch(rect)
            # Show r2,c2
            rect = patches.Rectangle((c2-0.5,r2-0.5),1,1,linewidth=2,edgecolor='b',facecolor='none')
            ax.add_patch(rect)
            ax.text(c1-0.3,r1+0.4,idx, color='yellow')
            ax.text(c2-0.3,r2+0.4,idx, color='yellow')
        # Display the merged result
        plt.show()

    return predictors

def loadData (which):
    faces = np.load("{}ingFaces.npy".format(which))
    faces = faces.reshape(-1, 24, 24)  # Reshape from 576 to 24x24
    labels = np.load("{}ingLabels.npy".format(which))
    return faces, labels

if __name__ == "__main__":
    testingFaces, testingLabels = loadData("test")
    trainingFaces, trainingLabels = loadData("train")
    
    numbers = [400, 800, 1200, 1600, 2000]
    print("n", "trainingAccuracy", "testingAccuracy", sep="\t")
    for n in numbers:
        predictors = stepwiseRegression(trainingFaces, trainingLabels, testingFaces, testingLabels, n)
        trainingAccuracy = measureAccuracyOfPredictors(predictors, trainingFaces, trainingLabels)
        testingAccuracy = measureAccuracyOfPredictors(predictors, testingFaces, testingLabels)
        print(n, trainingAccuracy, testingAccuracy, sep="\t")
