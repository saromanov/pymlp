from pymlp import MLP
from sklearn.cross_validation import train_test_split
from sklearn.datasets import load_iris

iris = load_iris()
print(iris.data[55], iris.target[55])
mlp = MLP(iris.data, iris.target, 7,3)
mlp.train()
for i in range(len(iris.data)):
	print("Result: ", mlp.predict(iris.data[i]), " : ", iris.target[i])