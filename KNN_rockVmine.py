from sklearn import neighbors
from sklearn import datasets
from sklearn.cross_validation import train_test_split

knn = neighbors.KNeighborsClassifier()

iris = datasets.load_iris()

xTrain, xTest, yTrain, yTest = train_test_split(iris.data, iris.target, test_size=0.330, random_state=531)

knn.fit(xTrain, yTrain)

predict_y = knn.predict(xTest)
count = 0
for i in range(len(yTest)):
    if yTest[i] == predict_y[i]:
        count += 1

print float(count) / float(len(yTest))
