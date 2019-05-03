###########################################
#           TESTING ACCURACY              #
# Linear accuracy: 0.8601397611927121     #
# Non-linear accuracy: 0.8369373215617769 #
###########################################

import sklearn.svm
import sklearn.metrics
import numpy as np
import pandas

# Load data
d = pandas.read_csv('train.csv')
y = np.array(d.target)  # Labels
X = np.array(d.iloc[:,2:])  # Features

# Split into train/test folds
n = int(len(y)/2)
random_indices = np.random.permutation(2*n)

ytr = y[random_indices[:n]] # select first 100000 indices
yte = y[random_indices[n:]] # select last 100000 indices

Xtr = X[random_indices[:n], :] # select first 100000 indices
Xte = X[random_indices[n:], :] # select last 100000 indices

# Linear SVM
print("Start linear training")
svm_linear = sklearn.svm.LinearSVC(dual=False)
svm_linear.fit(Xtr, ytr)
yhat1 = svm_linear.decision_function(Xte)


# Non-linear SVM (polynomial kernel)
print("Start polynomial training")
BAG_SIZE = 2000
bags = int(n/BAG_SIZE)
yhat2 = np.zeros(n)
for i in range(bags):
    print("Poly training:", i)
    start = i * BAG_SIZE
    end = (i+1) * BAG_SIZE
    svm_poly = sklearn.svm.SVC(kernel='poly', degree=3, gamma='auto')
    svm_poly.fit(Xtr[start:end, :], ytr[start:end])
    yhat2 += svm_poly.decision_function(Xte)
yhat2 = yhat2 / bags

# Apply the SVMs to the test set
#yhat1 = ...  # Linear kernel
#yhat2 = ...  # Non-linear kernel

# Compute AUC
auc1 = sklearn.metrics.roc_auc_score(yte, yhat1)
auc2 = sklearn.metrics.roc_auc_score(yte, yhat2)

print("Linear accuracy:", auc1)
print("Non-linear accuracy:", auc2)