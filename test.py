from sklearn.datasets import make_hastie_10_2
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import *


data, target = make_hastie_10_2(400)
print(data.shape)
data[0,:]
print(data)
print(target)
data_train, data_test, target_train, target_test = train_test_split(data, target, test_size=0.2)
print("Training set has %d examples" % data_train.shape[0] )
print("Test set has %d examples" % data_test.shape[0] )


model = KNeighborsClassifier(n_neighbors=3)
model.fit(data_train, target_train)
print(model)

predicted = model.predict(data_test)
# print("Target",target_test)
# print("Predictions",predicted)

# build the confusion matrix
cm = confusion_matrix(target_test, predicted,labels=[1,-1])
print(cm)
print("Accuracy = %.2f" % accuracy_score(target_test, predicted) )