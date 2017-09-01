__author__ = 'mike_bowles'

import urllib2
from math import sqrt, fabs, exp
import matplotlib.pyplot as plot
from sklearn.cross_validation import train_test_split
from sklearn import ensemble
from sklearn.metrics import roc_auc_score, roc_curve
import numpy
from sklearn import svm

#read data from uci data repository
# target_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/sonar/sonar.all-data"
# data = urllib2.urlopen(target_url)

with open('sonar.all-data', 'r') as f:
    data = f.readlines()

#arrange data into list for labels and list of lists for attributes
xList = []

for line in data:
    #split on comma
    row = line.strip().split(",")
    xList.append(row)

#separate labels from attributes, convert from attributes from string to numeric and convert "M" to 1 and "R" to 0

xNum = []
labels = []

for row in xList:
    lastCol = row.pop()
    if lastCol == "M":
        labels.append(1)
    else:
        labels.append(0)
    attrRow = [float(elt) for elt in row]
    xNum.append(attrRow)

#number of rows and columns in x matrix
nrows = len(xNum)
ncols = len(xNum[1])

#form x and y into numpy arrays and make up column names
X = numpy.array(xNum)
y = numpy.array(labels)
rocksVMinesNames = numpy.array(['V' + str(i) for i in range(ncols)])

#break into training and test sets.
xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size=0.30, random_state=531)

clf = svm.SVC(kernel='poly', degree=4)
clf.fit(xTrain, yTrain)
predict_y = clf.predict(xTest)
count = 0
for i in range(len(yTest)):
    if yTest[i] == predict_y[i]:
        count += 1
print 'Accuracy: %f' % (float(count)/float(len(yTest)))


#
#
# auc = []
# nTreeList = range(50, 2000, 50)
# for iTrees in nTreeList:
#     depth = None
#     maxFeat = 8  # try tweaking
#     rocksVMinesRFModel = ensemble.RandomForestClassifier(n_estimators=iTrees, max_depth=depth, max_features=maxFeat,
#                                                  oob_score=False, random_state=531)
#
#     rocksVMinesRFModel.fit(xTrain, yTrain)
#
#     # Accumulate auc on test set
#     prediction = rocksVMinesRFModel.predict_proba(xTest)
#     aucCalc = roc_auc_score(yTest, prediction[:,1:2])
#     auc.append(aucCalc)
#
# print("AUC")
# print(auc[-1])
