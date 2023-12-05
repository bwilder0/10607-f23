# no imports beyond the ones below should be needed in answering this question
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

def sigmoid(z):
	"""sigmoid function"""
	result = 1 / (1 + np.exp(-z))

	assert np.all(0.0 <= result)

	assert np.all(result <= 1.0)

	return result

def log_loss(y, y_probs):
	"""log-loss function"""
	assert np.all(y_probs >= 0.0)

	assert np.all(y_probs <= 1.0)

	result = (-y * np.log(y_probs) - (1 - y) * np.log(1 - y_probs)).mean()

	return result 

def log_loss_grad(X, y, w):
	"""gradient of the log-loss function"""
	assert len(X.shape)==2

	assert len(y.shape)==1

	assert X.shape[1] == w.shape[0]

	assert X.shape[0] == y.shape[0]

	y_probs = predict_probs(X, w)

	assert y_probs.shape[0] == y.shape[0]

	result = np.dot(X.T, (y_probs - y)) / y.shape[0]

	return result

def predict_probs(X, w):
	"""predict logistic regression probabilities"""
	assert X.shape[1] == w.shape[0]

	result = sigmoid(np.dot(X, w))

	return result

def predict(X, w, threshold=0.5):
	"""make logistic regression predictions using a specified threshold
	to binarize the probability threshold
	"""
	return 1.0 * (predict_probs(X, w) >= threshold)

def evaluate_accuracy(X, y, w):
	"""evaluate accuracy by making predictions
	and comparing with groundtruth"""
	y_predict = predict(X, w)

	result = (y == y_predict).mean()

	assert (0.0 <= result <= 1.0)

	return result

def gradient_descent(w, X, y, f_grad, lr = 1e-2):
	"""makes an update using gradient descent
	where the gradient is calculated using all the data
	
	Parameters:
		w: current weight parameter, shape = num_features
		X: input features, shape = num_datapoints, num_features
		y: binary output target, shape = num_datapoints
		f_grad: a Python function which computes the gradient of the log-loss function, use log_loss_grad here
		lr: learning rate for gradient descent
	"""
	# Do not edit any code outside the edit region
	# Edit region starts here
	#########################
	# Your code goes here
	#########################
	# Edit region ends here

	assert X.shape[1] == w.shape[0]

def stochastic_gradient_descent(w, X, y, f_grad, lr = 1e-2):
	"""makes an update using stochastic gradient descent
	where the gradient is calculated using a randomly chosen datapoint

	Parameters:
		w: current weight parameter, shape = num_features
		X: input features, shape = num_datapoints, num_features
		y: binary output target, shape = num_datapoints
		f_grad: a Python function which computes the gradient of the log-loss function, use log_loss_grad here
		lr: learning rate for stochastic gradient descent
	"""
	# Do not edit any code outside the edit region
	# Edit region starts here
	#########################
	# Your code goes here
	#########################
	# Edit region ends here

	assert X.shape[1] == w.shape[0]

def adagrad(w, X, y, f_grad, gti, lr = 1e-2, eps_stable = 1e-8):
	"""makes an update using adagrad
	where the gradient is calculated using a randomly chosen datapoint
	and gti maintains the running sum of squared gradient magnitudes required for the adagrad update

	Parameters:
		w: current weight parameter, shape = num_features
		X: input features, shape = num_datapoints, num_features
		y: binary output target, shape = num_datapoints
		f_grad: a Python function which computes the gradient of the log-loss function, use log_loss_grad here
		gti: maintains the running sum of squared gradient magnitude, shape = num_features
		lr: learning rate for AdaGrad
	"""
	# Do not edit any code outside the edit region
	# Edit region starts here
	#########################
	# Your code goes here
	#########################
	# Edit region ends here

	assert X.shape[1] == w.shape[0]

if __name__ == '__main__':
	# set numpy seed for reproducibility
	np.random.seed(666)

	# load well-known Iris dataset from scikit-learn package
	# convert from 3 to 2 classes for binary classification
	iris = load_iris()
	X = iris.data[:, :2]
	y = (iris.target != 0) * 1

	# add a constant feature at position 0 of datapoints
	# the first weight therefore corresponds to the bias term
	intercept = np.ones((X.shape[0], 1))
	X = np.concatenate((intercept, X), axis=1)

	# split data into train and test using a scikit-learn utility function
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=666)

	# number of epochs
	num_epochs = 10000

	# initialize weights for gradient descent, stochastic gradient descent, adagrad
	w_gd = np.zeros(X.shape[1])
	w_sgd = np.zeros(X.shape[1])
	w_adagrad = np.zeros(X.shape[1])

	# initialize G_tii for adagrad
	gti = np.zeros(X.shape[1])

	for epoch in range(num_epochs):
		# gradient descent is executed only once per epoch since it runs over entire data
		gradient_descent(w_gd, X_train, y_train, log_loss_grad)

		for i in range(X_train.shape[0]):
			# stochastic gradient descent and adagrad executed as many times in an epoch as the number of datapoints
			stochastic_gradient_descent(w_sgd, X_train, y_train, log_loss_grad)
			adagrad(w_adagrad, X_train, y_train, log_loss_grad, gti)

		# calculate and print train logistic loss using the three update methods at the end of each epoch
		train_logloss_gd = log_loss(y_train, predict_probs(X_train, w_gd))
		train_logloss_sgd = log_loss(y_train, predict_probs(X_train, w_sgd))
		train_logloss_adagrad = log_loss(y_train, predict_probs(X_train, w_adagrad))
		print('Train LogLoss (GD, SGD, AdaGrad):', train_logloss_gd, train_logloss_sgd, train_logloss_adagrad)

		# calculate and print train accuracies using the three update methods at the end of each epoch
		train_accuracy_gd = evaluate_accuracy(X_train, y_train, w_gd)
		train_accuracy_sgd = evaluate_accuracy(X_train, y_train, w_sgd)
		train_accuracy_adagrad = evaluate_accuracy(X_train, y_train, w_adagrad)
		print('Train Accuracies (GD, SGD, AdaGrad):', train_accuracy_gd, train_accuracy_sgd, train_accuracy_adagrad)

	# calculate test accuracies using the three update methods
	test_accuracy_gd = evaluate_accuracy(X_test, y_test, w_gd)
	test_accuracy_sgd = evaluate_accuracy(X_test, y_test, w_sgd)
	test_accuracy_adagrad = evaluate_accuracy(X_test, y_test, w_adagrad)
	print('Test Accuracies (GD, SGD, AdaGrad):', test_accuracy_gd, test_accuracy_sgd, test_accuracy_adagrad)
