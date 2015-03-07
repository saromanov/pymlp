import numpy as np
from sklearn import datasets

def sigmoid(x):
	return 1.0/(1.0 + np.exp(-x))

class MLP:
	def __init__(self, x, y, num_hid, num_out, Wih=None, Woh=None):
		self.x = x
		self.y = y
		self.Wih = np.random.random((x.shape[1], num_hid))
		self.Woh = np.random.random((num_hid, num_out))
		self.bias = np.random.random(num_hid)

	def forward(self, item=None):
		inp = item
		if item == None:
			inp = self.x
		hidden = sigmoid(np.dot(inp, self.Wih) + self.bias)
		output = sigmoid(np.dot(hidden, self.Woh))
		return hidden, output

	def backward(self, hidden, output,beta=0.05):
		out_fun = np.zeros((self.y.shape[0],1))
		for i in range(self.y.shape[0]):
			out_fun[i] = self.y[i] - output[i]
		output_error = beta * np.multiply(out_fun, np.multiply(output, 1 - output))
		hidden_check = beta * np.multiply(hidden,  (1 - hidden))
		dotter = np.dot(output_error, self.Woh.T)
		hidden_error = np.multiply(hidden_check, dotter)
		return hidden_error, output_error

	def _error(self, output):
		return 1/(self.y.shape[0]) * np.sum(np.sqrt((output - self.y)**2))

	def gradient_step(self, hidden_error, output_error, hidden, output, lrate):
		return lrate * np.dot(hidden_error.T, self.x), \
			   lrate * np.dot(output_error.T, hidden)


	def train(self, lrate=0.001, iters=10):
		for i in range(500):
			hidden, output = self.forward()
			hidden_error, output_error = self.backward(hidden, output)
			Wih_new, Who_out = self.gradient_step(hidden_error, output_error, hidden, output, lrate)
			self.Wih -= Wih_new.T
			self.Woh -= Who_out.T

	def predict(self, values):
		hidden, output = self.forward(values)
		return np.argmax(output)
