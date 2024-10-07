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
		features = iris[:, :-1]
		return np.mean(features, axis=0)

	def empirical_covariance(self, iris):
		features = iris[:, :-1]  # Exclude the last column (class labels)
		return np.cov(features, rowvar=False)

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


def manhattan_distance(x, p):
	return np.sum(np.abs(x - p))


class HardParzen:
	def __init__(self, h):
		self.label_list = None
		self.h = h
		self.train_inputs = None
		self.train_labels = None
		self.n_classes = None

	def fit(self, train_inputs, train_labels):
		self.label_list = np.unique(train_labels)
		self.train_inputs = train_inputs
		self.train_labels = train_labels
		self.n_classes = len(self.label_list)


	def predict(self, test_data):
		class_predictions = np.zeros(test_data.shape[0])

		for i, current_point in enumerate(test_data):
			count = np.zeros(self.n_classes)

			for j, train_point in enumerate(self.train_inputs):
				distance = manhattan_distance(current_point, train_point)
				if distance < self.h:
					label = int(self.train_labels[j])
					count[label - 1] += 1

			if np.sum(count) == 0:
				class_predictions[i] = draw_rand_label(current_point, self.label_list)
			else:
				class_predictions[i] = np.argmax(count) + 1

		return class_predictions


class SoftRBFParzen:
	def __init__(self, sigma):
		self.sigma  = sigma
		self.label_list = None
		self.train_inputs = None
		self.train_labels = None
		self.n_classes = None

	def fit(self, train_inputs, train_labels):
		self.label_list = np.unique(train_labels)
		self.train_inputs = train_inputs
		self.train_labels = train_labels
		self.n_classes = len(self.label_list)

	def rbf_kernel(self, distance):
		return np.exp(- (distance ** 2) / (2 * self.sigma ** 2))

	def predict(self, test_data):
		class_predictions = np.zeros(test_data.shape[0])

		for i, current_point in enumerate(test_data):
			weights = np.zeros(self.n_classes)

			for j, train_point in enumerate(self.train_inputs):
				distance = manhattan_distance(current_point, train_point)
				current_weight = self.rbf_kernel(distance)
				current_label = int(self.train_labels[j])
				weights[current_label - 1] += current_weight

			class_predictions[i] = np.argmax(weights) + 1

		return class_predictions


def split_dataset(iris):
	training_set = []
	validation_set = []
	test_set = []

	for i in range(len(iris)):
		if i % 5 in [0, 1, 2]:
			training_set.append(iris[i])
		elif i % 5 == 3:
			validation_set.append(iris[i])
		elif i % 5 == 4:
			test_set.append(iris[i])

	training_set = np.array(training_set)
	validation_set = np.array(validation_set)
	test_set = np.array(test_set)

	result = (training_set, validation_set, test_set)
	return result


class ErrorRate:
	def __init__(self, x_train, y_train, x_val, y_val):
		self.x_train = x_train
		self.y_train = y_train
		self.x_val = x_val
		self.y_val = y_val

	def hard_parzen(self, h):
		hp = HardParzen(h)
		hp.fit(self.x_train, self.y_train)
		predictions = hp.predict(self.x_val)
		error_rate = np.mean(predictions != self.y_val)
		return error_rate



	def soft_parzen(self, sigma):
		sp = SoftRBFParzen(sigma)
		sp.fit(self.x_train, self.y_train)
		predictions = sp.predict(self.x_val)
		error_rate = np.mean(predictions != self.y_val)
		return error_rate


def get_test_errors(iris):
	pass


def random_projections(X, A):
	pass


training_set, validation_set, test_set = split_dataset(iris)
x_train = training_set[:, :-1]
y_train = training_set[:, -1]
x_val = validation_set[:, :-1]
y_val = validation_set[:, -1]
error_rate = ErrorRate(x_train, y_train, x_val, y_val)
sp_error_rate = error_rate.soft_parzen(0.6)
print(sp_error_rate)

