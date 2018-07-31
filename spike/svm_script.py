from sklearn import datasets
from sklearn import svm

import matplotlib.pyplot as plt

digits = datasets.load_digits()

clf = svm.SVC(gamma=.0001, C=100)

print(len(digits.data))

x, y = digits.data[:-10], digits.target[:-10]

print(digits.target[:20])

#clf.fit(x, y)

#print("Prediction:", clf.predict(digits.data[-6]))
plt.imshow(digits.images[1], cmap=plt.cm.gray_r, interpolation="nearest")
plt.show()