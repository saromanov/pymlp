import numpy as np
from sklearn import datasets

def sigmoid(x):
	return 1/(1 + np.exp(-x))

class MLP:
	def __init__(self, x, y, num_hid, Wih=None, Woh=None):
		self.x = x
		self.y = y
		self.Wih = np.random.random((x.shape[1], num_hid))
		self.Woh = np.random.random((num_hid, 1))
		self.bias = np.random.random(num_hid)

	def forward(self):
		hidden = np.dot(self.x, self.Wih) + self.bias
		output = np.dot(hidden, self.Woh)
		return hidden, output

	def backward(self, hidden, output):
		out_fun = np.arange(self.y.shape[0])
		for i in range(self.y.shape[0]):
			out_fun[i] = self.y[i] - output[i]
		output_error = out_fun *  np.dot(np.transpose(output), 1 - output)
		hidden_check = hidden * (1 - hidden)
		hidden_error = np.dot(hidden_check, np.dot(self.Woh, output_error))
		return hidden_error, output_error

	def gradient_step(self, hidden_error, output_error, hidden, output, lrate):
		return lrate * np.dot(output_error, hidden), \
			   lrate * np.dot(hidden_error, output)


	def train(self, lrate=0.01, iters=10):
		for i in range(10):
			hidden, output = self.forward()
			hidden_error, output_error = self.backward(hidden, output)
			Wih_new, Who_out = self.gradient_step(hidden_error, output_error, hidden, output, lrate)
			self.Wih -= Wih_new
			self.Woh -= Who_out


iris = datasets.load_iris()
mlp = MLP(iris.data, iris.target, 2)
mlp.train()