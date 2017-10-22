import numpy as np

inputLayerSize, hiddenLayerSize, outputLayerSize = 2, 3, 1

learningRate = 0.01

epoch = 20000

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y = np.array([[0], [1], [1], [0]])

weight_input_hidden = np.random.uniform(size=(inputLayerSize, hiddenLayerSize))
weight_hidden_output = np.random.uniform(size=(hiddenLayerSize, outputLayerSize))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


for i in range(epoch):
    activation_hidden = sigmoid(np.dot(X, weight_input_hidden))
    activation_output = np.dot(activation_hidden, weight_hidden_output)
    error = Y - activation_output
    derivative_output = error * learningRate
    weight_hidden_output += activation_hidden.T.dot(derivative_output)
    derivative_hidden = derivative_output.dot(weight_hidden_output.T) * sigmoid_derivative(activation_hidden)
    weight_input_hidden += X.T.dot(derivative_hidden)

print(activation_output)

