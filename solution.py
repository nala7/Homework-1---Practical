import numpy as np
iris = np.genfromtxt('iris.txt')

######## DO NOT MODIFY THIS FUNCTION ########
def draw_rand_label(x, label_list):
	seed = abs(np.sum(x))
	while seed < 1:
		seed = 10 * seed
	seed = int(1000000 * seed)
	np.random.seed(seed)
	return np.random.choice(label_list)
#############################################


class Q1:
	def feature_means(self, iris):
		return np.mean(iris.data, axis=0)

	def empirical_covariance(self, iris):
		return np.cov(iris.data, rowvar=False)

	def feature_means_class_1(self, iris):
		features = iris[:, :-1]
		labels = iris[:, -1]

		class_1_mask = labels == 1
		class_1_data = features[class_1_mask]
		return np.mean(class_1_data, axis=0)

	def empirical_covariance_class_1(self, iris):
		features = iris[:, :-1]
		labels = iris[:, -1]
		class_1_mask = labels == 1
		class_1_data = features[class_1_mask]
		return np.cov(class_1_data, rowvar=False)


class HardParzen:
	def __init__(self, h):
		self.h = h

	def fit(self, train_inputs, train_labels):
		# self.label_list = np.unique(train_labels)
		pass

	def predict(self, test_data):
		pass


class SoftRBFParzen:
	def __init__(self, sigma):
		self.sigma  = sigma

	def fit(self, train_inputs, train_labels):
		# self.label_list = np.unique(train_labels)
		pass

	def predict(self, test_data):
		pass


def split_dataset(iris):
	pass


class ErrorRate:
	def __init__(self, x_train, y_train, x_val, y_val):
		self.x_train = x_train
		self.y_train = y_train
		self.x_val = x_val
		self.y_val = y_val

	def hard_parzen(self, h):
		pass

	def soft_parzen(self, sigma):
		pass


def get_test_errors(iris):
	pass


def random_projections(X, A):
	pass


q1 = Q1()

print(q1.empirical_covariance_class_1(iris))