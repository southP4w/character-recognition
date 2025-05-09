import time
import matplotlib.pyplot as plt
import numpy as np

from random import randint
from matplotlib.colors import PowerNorm
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


class SVMClassifier:
    """
    Support Vector Machine (SVC) classifier in a one-vs-all setup.
    """
    def __init__(self, regularization_rate=10.579996, learning_rate=1e-3, number_of_epochs=100):
        """
        Parameterized constructor.

        :param regularization_rate (float): Determines the regularization strength. Larger value means less penalty on the larger weights.
        :param learning_rate (float): The step size for each gradient update.
        :param number_of_epochs (int): How many passes (epochs) over the training data.
        """
        self.regularization_rate    = regularization_rate
        self.learning_rate          = learning_rate
        self.number_of_epochs       = number_of_epochs
        self.weights                = []    # will become an array of shape (number_of_classes, number_of_features)
        self.biases                 = []    # will become an array of shape (number_of_classes)


    def fit(self, X, y):
        """
        Trains the SVC model on data X (in our case pixels) and labels y.

        :param X: Sample set
        :param y: Labels array
        """
        classes         = np.unique(y)  # get the unique class labels
        self.weights    = np.zeros((len(classes), X.shape[1]))  # initialize weight matrix: one weight vector per class
        self.biases     = np.zeros(len(classes))    # initialize bias vector: one bias per class
        for class_index, class_label in enumerate(classes): # Train a separate binary SVC for each class
            y_binary        = np.where(y==class_label, 1, -1)   # 1 for this class, -1 otherwise
            weight_vector   = np.zeros(X.shape[1])  #
            bias = 0.0
            for _ in range(self.number_of_epochs):  # gradient descent loop
                margins         = y_binary * (X.dot(weight_vector) + bias)  # compute the margin
                misclassified   = margins < 1   # identify the margin-violating samples
                # hinge-loss gradient + regularization:
                weight_gradient = weight_vector - self.regularization_rate*np.dot(y_binary[misclassified], X[misclassified])
                bias_gradient   = -self.regularization_rate*np.sum(y_binary[misclassified])
                weight_vector   -= self.learning_rate*weight_gradient   # update the weight vector
                bias            -= self.learning_rate*bias_gradient     # update the bias
            # store the updated/"learned" parameters for this class
            self.weights[class_index]   = weight_vector
            self.biases[class_index]    = bias


    def decision_function(self, X):
        """
        Computes the raw scores for each class.

        :param X: Sample set
        :return: Weight scores matrix (number_of_samples, number_of_features)
        """
        return X.dot(self.weights.T) + self.biases


    def predict(self, X):
        """
        Predicts the class labels for the samples in X.
        :param X: Sample set
        :return: Array of predictions
        """
        return np.argmax(self.decision_function(X), axis=1)


    def score(self, X, y):
        """

        :param X: Sample set
        :param y: Labels
        :return: Accuracy score
        """
        return np.mean(self.predict(X) == y)


def gaussian_naive_bayes_scikit(X_train, y_train, X_test, y_test, display_conf_matx: bool):
    classifier  = GaussianNB()
    classifier.fit(X_train, y_train)
    predictions = classifier.predict(X_test)
    acc_score   = accuracy_score(y_test, predictions)
    conf_matx   = confusion_matrix(y_test, predictions)
    clas_rprt   = classification_report(y_test, predictions)
    with open('results.txt', 'a') as output_file:
        print('\n==== Gaussian Naive Bayes ====', sep=' ', end='\n', flush=False, file=output_file)
        print('Confusion Matrix:\n', conf_matx, sep=' ', end='\n', flush=False, file=output_file)
        print('Classification Report:\n', clas_rprt, sep=' ', end='\n', flush=False, file=output_file)
        print('Accuracy:', acc_score, sep=' ', end='\n', flush=False, file=output_file)
        print('=========================================', sep=' ', end='\n', flush=False, file=output_file)
    if display_conf_matx:
        fig = plot_confusion_matrix(y_test, predictions, classes=range(10))
        fig.suptitle(f'Gaussian NB Confusion Matrix Heatmap')
        plt.title(f'Accuracy: {acc_score: .5f}')
        plt.show()

    return classifier


def k_nearest_neighbors_scikit(k: int, X_train, y_train, X_test, y_test, display_conf_matx: bool):
    classifier  = KNeighborsClassifier(n_neighbors=k)
    classifier.fit(X_train, y_train)
    predictions = classifier.predict(X_test)
    acc_score   = accuracy_score(y_test, predictions)
    conf_matx   = confusion_matrix(y_test, predictions)
    clas_rprt   = classification_report(y_test, predictions)
    with open('results.txt', 'a') as output_file:
        print('\n==== K-Nearest Neighbors ====', sep=' ', end='\n', flush=False, file=output_file)
        print('Confusion Matrix:\n', conf_matx, sep=' ', end='\n', flush=False, file=output_file)
        print('Classification Report:\n', clas_rprt, sep=' ', end='\n', flush=False, file=output_file)
        print('Accuracy:', acc_score, sep=' ', end='\n', flush=False, file=output_file)
        print('=========================================', sep=' ', end='\n', flush=False, file=output_file)
    if display_conf_matx:
        fig = plot_confusion_matrix(y_test, predictions, classes=range(10))
        fig.suptitle(f'KNN Confusion Matrix Heatmap')
        plt.title(f'k (Number of Neighbors): {k} | Accuracy: {acc_score: .5f}')
        plt.show()

    return classifier


def logistic_regression_scikit(X_train, y_train, X_test, y_test, display_conf_matx: bool):
    classifier  = LogisticRegression(
            max_iter=1000,
            solver='saga',
            random_state=int(time.time()) + randint(0, int(time.time()))
    )
    classifier.fit(X_train, y_train)
    predictions = classifier.predict(X_test)
    acc_score   = accuracy_score(y_test, predictions)
    conf_matx   = confusion_matrix(y_test, predictions)
    clas_rprt   = classification_report(y_test, predictions)
    with open('results.txt', 'a') as output_file:
        print('\n==== Logistic Regression ====', sep=' ', end='\n', flush=False, file=output_file)
        print('Confusion Matrix:\n', conf_matx, sep=' ', end='\n', flush=False, file=output_file)
        print('Classification Report:\n', clas_rprt, sep=' ', end='\n', flush=False, file=output_file)
        print('Accuracy:', acc_score, sep=' ', end='\n', flush=False, file=output_file)
        print('=========================================', sep=' ', end='\n', flush=False, file=output_file)
    if display_conf_matx:
        fig = plot_confusion_matrix(y_test, predictions, classes=range(10))
        fig.suptitle(f'LogReg Confusion Matrix Heatmap')
        plt.title(f'Accuracy: {acc_score: .5f}')
        plt.show()

    return classifier


def support_vector_machine(C, X_train, y_train, X_test, y_test, display_conf_matx: bool):
    classifier  = SVMClassifier(regularization_rate=C)
    classifier.fit(X_train, y_train)
    predictions = classifier.predict(X_test)
    acc_score   = accuracy_score(y_test, predictions)
    conf_matx   = confusion_matrix(y_test, predictions)
    clas_rprt   = classification_report(y_test, predictions)
    with open('results.txt', 'a') as output_file:
        print('\n==== (My) Support Vector Machine ====', sep=' ', end='\n', flush=False, file=output_file)
        print('Confusion Matrix:\n', conf_matx, sep=' ', end='\n', flush=False, file=output_file)
        print('Classification Report:\n', clas_rprt, sep=' ', end='\n', flush=False, file=output_file)
        print('Accuracy:', acc_score, sep=' ', end='\n', flush=False, file=output_file)
        print('=========================================', sep=' ', end='\n', flush=False, file=output_file)
    if display_conf_matx:
        fig = plot_confusion_matrix(y_test, predictions, classes=range(10))
        fig.suptitle('(My) SVC Confusion Matrix Heatmap')
        plt.title(f'C (Penalty): {C} | Accuracy: {acc_score:.5f}')
        plt.show()

    return classifier


def support_vector_machine_scikit(C, X_train, y_train, X_test, y_test, display_conf_matx: bool):
    classifier  = SVC(
            C=C,
            probability=True
    )
    classifier.fit(X_train, y_train)
    predictions = classifier.predict(X_test)
    acc_score   = accuracy_score(y_test, predictions)
    conf_matx   = confusion_matrix(y_test, predictions)
    clas_rprt   = classification_report(y_test, predictions)
    with open('results.txt', 'a') as output_file:
        print('\n==== Support Vector Machine ====', sep=' ', end='\n', flush=False, file=output_file)
        print('Confusion Matrix:\n', conf_matx, sep=' ', end='\n', flush=False, file=output_file)
        print('Classification Report:\n', clas_rprt, sep=' ', end='\n', flush=False, file=output_file)
        print('Accuracy:', acc_score, sep=' ', end='\n', flush=False, file=output_file)
        print('=========================================', sep=' ', end='\n', flush=False, file=output_file)
    if display_conf_matx:
        fig = plot_confusion_matrix(y_test, predictions, classes=range(10))
        fig.suptitle(f'SVC Confusion Matrix Heatmap')
        plt.title(f'C (Penalty): {C} | Accuracy: {acc_score: .5f}')
        plt.show()

    return classifier


def decision_tree_scikit(max_depth, X_train, y_train, X_test, y_test, display_conf_matx: bool):
    classifier  = DecisionTreeClassifier(
            max_depth=max_depth,
            max_features=None,
            random_state=int(time.time()) + randint(0, int(time.time()))
    )
    classifier.fit(X_train, y_train)
    predictions = classifier.predict(X_test)
    acc_score   = accuracy_score(y_test, predictions)
    conf_matx   = confusion_matrix(y_test, predictions)
    clas_rprt   = classification_report(y_test, predictions)
    with open('results.txt', 'a') as output_file:
        print('\n==== Decision Tree ====', sep=' ', end='\n', flush=False, file=output_file)
        print('Confusion Matrix:\n', conf_matx, sep=' ', end='\n', flush=False, file=output_file)
        print('Classification Report:\n', clas_rprt, sep=' ', end='\n', flush=False, file=output_file)
        print('Accuracy:', acc_score, sep=' ', end='\n', flush=False, file=output_file)
        print('=========================================', sep=' ', end='\n', flush=False, file=output_file)
    if display_conf_matx:
        fig = plot_confusion_matrix(y_test, predictions, classes=range(10))
        fig.suptitle(f'DTC Confusion Matrix Heatmap')
        plt.title(f'Max Depth: {max_depth} | Accuracy: {acc_score: .5f}')
        plt.show()

    return classifier


def random_decision_forest_scikit(n: int, X_train, y_train, X_test, y_test, display_conf_matx: bool):
    classifier  = RandomForestClassifier(
            n_estimators=n,
            random_state=int(time.time()) + randint(0, int(time.time()))    # time-based seed
    )
    classifier.fit(X_train, y_train)
    predictions = classifier.predict(X_test)
    acc_score   = accuracy_score(y_test, predictions)
    conf_matx   = confusion_matrix(y_test, predictions)
    clas_rprt   = classification_report(y_test, predictions)
    with open('results.txt', 'a') as output_file:
        print('\n==== Random Decision Forest ====', sep=' ', end='\n', flush=False, file=output_file)
        print('Confusion Matrix:\n', conf_matx, sep=' ', end='\n', flush=False, file=output_file)
        print('Classification Report:\n', clas_rprt, sep=' ', end='\n', flush=False, file=output_file)
        print('Accuracy:', acc_score, sep=' ', end='\n', flush=False, file=output_file)
        print('=========================================', sep=' ', end='\n', flush=False, file=output_file)
    if display_conf_matx:
        fig = plot_confusion_matrix(y_test, predictions, classes=range(10))
        fig.suptitle(f'RDF Confusion Matrix Heatmap')
        plt.title(f'Estimators: {n} | Accuracy: {acc_score: .5f}')
        plt.show()

    return classifier


def multi_level_perceptron_scikit(X_train, y_train, X_test, y_test, display_conf_matx: bool):
    classifier  = MLPClassifier(
        hidden_layer_sizes=15,
        max_iter=1000,
        random_state=int(time.time()) + randint(0, int(time.time()))
    )
    classifier.fit(X_train, y_train)
    predictions = classifier.predict(X_test)
    acc_score   = accuracy_score(y_test, predictions)
    conf_matx   = confusion_matrix(y_test, predictions)
    clas_rprt   = classification_report(y_test, predictions)
    with open('results.txt', 'a') as output_file:
        print('\n==== Multi-Layer Perceptron ====', sep=' ', end='\n', flush=False, file=output_file)
        print('Confusion Matrix:\n', conf_matx, sep=' ', end='\n', flush=False, file=output_file)
        print('Classification Report:\n', clas_rprt, sep=' ', end='\n', flush=False, file=output_file)
        print('Accuracy:', acc_score, sep=' ', end='\n', flush=False, file=output_file)
        print('=========================================', sep=' ', end='\n', flush=False, file=output_file)
    if display_conf_matx:
        fig = plot_confusion_matrix(y_test, predictions, classes=range(10))
        fig.suptitle(f'MLP Confusion Matrix Heatmap')
        plt.title(f'Layers: {classifier.hidden_layer_sizes} | Accuracy: {acc_score: .5f}')
        plt.show()

    return classifier


def plot_confusion_matrix(y_true, y_predicted, classes):
    fig, axes   = plt.subplots()
    conf_matx   = confusion_matrix(y_true, y_predicted)
    image       = axes.imshow(conf_matx, cmap='Blues', norm=PowerNorm(gamma=.33))
    fig.colorbar(image, ax=axes)
    axes.set_xticks(np.arange(len(classes)))
    axes.set_yticks(np.arange(len(classes)))
    axes.set_xticklabels(classes)
    axes.set_yticklabels(classes)
    plt.setp(axes.get_xticklabels(), rotation=45, ha='right')
    axes.set_ylabel('True Label')
    axes.set_xlabel('Predicted Label')

    return fig
