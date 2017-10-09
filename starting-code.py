import numpy as np
import matplotlib.pyplot as plt


def to_categorical (y, num_classes=None):
    y = np.array(y, dtype='int').ravel()
    if not num_classes: num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes))
    categorical[np.arange(n), y] = 1
    return categorical


def load_datasets (files_path: str):
	from pandas import read_csv
	train_x = read_csv("{}csvTrainImages 13440x1024.csv".format(files_path), header=None).values.astype('float32').reshape([-1, 32, 32, 1]) / 255.0
	train_y = read_csv("{}csvTrainLabel 13440x1.csv".format(files_path), header=None).values.astype('int32') - 1
	test_x = read_csv("{}csvTestImages 3360x1024.csv".format(files_path), header=None).values.astype('float32').reshape([-1, 32, 32, 1]) / 255.0
	test_y = read_csv("{}csvTestLabel 3360x1.csv".format(files_path), header=None).values.astype('int32') - 1
	return train_x, train_y, test_x, test_y


def accuracy (y_true: np.ndarray, y_pred: np.ndarray):
	# ======= Just some sanity checks ========
	assert isinstance(y_true, np.ndarray)
	assert isinstance(y_pred, np.ndarray)
	assert len(y_true.shape) == 2
	assert len(y_pred.shape) == 2
	assert y_true.shape[0] == 28
	assert y_pred.shape[0] == 28
	assert y_true.shape[1] == y_pred.shape[1]
	# ======= All systems are go! ============

	results = np.zeros((28,))
	for i in range(y_true.shape[1]):
		results += np.logical_and(y_true[:, i] > 0.5, y_pred[:, i] > 0.5)

	summation = np.sum(y_true, axis=1)
	assert summation.shape[0] == 28

	accuracy_per_class = (results / summation) * 100.0
	overall_accuracy = np.mean(accuracy_per_class)

	return overall_accuracy, accuracy_per_class


def plot_randomly (training_set: np.ndarray, training_labels: np.ndarray, testing_set: np.ndarray, testing_labels: np.ndarray):
	from random import randint

	arabic_labels = [
		'alef', 'beh', 'teh', 'theh', 'jeem', 'hah', 'khah', 'dal', 'thal',
		'reh', 'zah', 'seen', 'sheen', 'sad', 'dad', 'tah', 'zah', 'ain',
		'ghain', 'feh', 'qaf', 'kaf', 'lam', 'meem', 'noon', 'heh', 'waw', 'yeh',
	]

	f, axarr = plt.subplots(4, 4)
	subplots = []
	for i in range(axarr.shape[0]):
		for j in range(axarr.shape[1]):
			subplots.append(axarr[i, j])

	for sp_index, subplot in enumerate(subplots):
		if sp_index < int(len(subplots) / 2):
			subplot_dataset = "training"
			random_index = randint(0, training_set.shape[0] - 1)
			subplot_image = training_set[random_index]
			subplot_class = training_labels[random_index][0]
		else:
			subplot_dataset = "testing"
			random_index = randint(0, testing_set.shape[0] - 1)
			subplot_image = testing_set[random_index]
			subplot_class = testing_labels[random_index][0]

		subplot_title = "Image #{}, {} set, class {}".format(random_index, subplot_dataset, arabic_labels[subplot_class])
		subplot.imshow(subplot_image.squeeze().T, cmap='gray')
		subplot.axis('off')
		subplot.set_title(subplot_title, size=8)

	plt.show()


def train_model (training_set: np.ndarray, training_labels: np.ndarray, testing_set: np.ndarray, testing_labels: np.ndarray):
	pass


if __name__ == "__main__":
	'''
	Welcome to the JOSA Deep Learning study groups first homework! In this homework, we'll be
	classifying handwritten Arabic letters (all 28 of them) using whatever neural network you want to build.

	Start by loading in your training + testing datasets using the load_datasets function. The
	load_datasets function takes a path as its argument. In the folder located by the path, Python should find
	four CSV files with their original names.
	'''
	train, train_labels, test, test_labels = load_datasets("path/to/your/datasets")

	'''
	Always make sure the shape of your dataset is correct! Your output should look like this:

		(13440, 32, 32, 1)
		(13440, 1)
		(3360, 32, 32, 1)
		(3360, 1)
	'''
	print(*[x.shape for x in [train, train_labels, test, test_labels]], sep="\n")

	'''
	What does our dataset look like? The plot_randomly function will do some matplotlib magic to randomly
	display 4 images from the training set, and 4 images from the testing set. Above each image, you will
	see the index of that image, the set it was taken from (training or testing), and most importantly the
	class of the set. Our alphabet will be sorted by the "Hijā’ī" sorting, so "alef" will come first and
	"yeh" will come last.
	'''
	plot_randomly(train, train_labels, test, test_labels)

	'''
	For reference, here's the naming we will use for each letter. The class numbers are therefore the
	indices of this list. For example, "alef" belongs to class 0, and "yeh" belongs to class 27.
	'''
	arabic_labels = [
		'alef', 'beh', 'teh', 'theh', 'jeem', 'hah', 'khah', 'dal', 'thal',
		'reh', 'zah', 'seen', 'sheen', 'sad', 'dad', 'tah', 'zah', 'ain',
		'ghain', 'feh', 'qaf', 'kaf', 'lam', 'meem', 'noon', 'heh', 'waw', 'yeh',
	]

	'''
	More basic exploratory analysis of our datasets. Let's count the number of unique train/test labels. We
	should get exactly 28. This is useful later when we build the output layer of our neural network.
	'''
	unique_classes = list(set(train_labels.squeeze().tolist() + test_labels.squeeze().tolist()))
	number_of_outputs = len(unique_classes)
	assert number_of_outputs == 28

	'''
	One last important step: converting our output into a big binary vector. Remember that for classification
	problems, we will output the probability that an input belongs to one specific class. When we have multiple
	classes, we need an output for each class probability. The to_categorical function simply takes each
	label and converts it into a binary vector, something like [0, 0, ... 1, 0, 0]. The 1 in that vector is in
	the same index as the corresponding label is in the arabic_labels list. You can check that out for yourself
	by printing train_labels[0].
	'''
	train_labels, test_labels = to_categorical(train_labels, number_of_outputs), to_categorical(test_labels, number_of_outputs)

	'''
	Show me what you got! I've left the train_model function empty for you, so you can experiment with
	the train set, build your neural network, and try it out on the test set. Our only condition is that you
	use the accuracy(y_true, y_pred) function to measure your accuracy on the train/test sets. Check that function
	out to see what it takes as input and what it returns as output. Happy coding!
	'''
	train_model(train, train_labels, test, test_labels)
