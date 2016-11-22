import optunity
import sklearn
from sklearn.decomposition import PCA
from sklearn.metrics import precision_score

_SVM_SEARCH_SPACE = {
    'kernel': {
        'linear': {'C': [0, 2]},
        'rbf': {'log_gamma': [-5, 0], 'C': [0, 10]},
        'poly': {'degree': [2, 5], 'C': [0, 5], 'coef0': [0, 2]}
    }
}


def _train_model(x_train, y_train, kernel, C, log_gamma, degree, coef0):
    """A generic SVM training function, with arguments based on the chosen kernel."""
    if kernel == 'linear':
        model = sklearn.svm.SVC(kernel=kernel, C=C, class_weight='balanced')
    elif kernel == 'poly':
        model = sklearn.svm.SVC(kernel=kernel, C=C, degree=degree, coef0=coef0, class_weight='balanced')
    elif kernel == 'rbf':
        model = sklearn.svm.SVC(kernel=kernel, C=C, gamma=10 ** log_gamma, class_weight='balanced')
    else:
        raise ValueError("Unknown kernel function: %s" % kernel)
    model.fit(x_train, y_train)
    return model


def svm_tuned_precision(x_train, y_train, x_test, y_test, kernel='linear', C=0, log_gamma=0, degree=0, coef0=0):
    model = _train_model(x_train, y_train, kernel, C, log_gamma, degree, coef0)
    predictions = model.predict(x_test)
    return precision_score(y_test, predictions, average='weighted')


class FaceDetector(object):

    def __init__(self, features_data, labels, images_width, images_height, n_eigenvectors=150, **kwargs):
        self._features_data = features_data
        self._labels = labels
        self._images_width = images_width
        self._images_height = images_height
        self._n_eigenvectors = n_eigenvectors
        self._model = None
        self._pca = PCA(n_components=self._n_eigenvectors, svd_solver=kwargs.get('svd_solver', 'randomized'), whiten=True)
        self._config = kwargs

    def train(self):
        self._pca.fit(self._features_data)
        features_pca = self._pca.transform(self._features_data)

        cv_decorator = optunity.cross_validated(x=features_pca, y=self._labels, num_folds=5)

        svm_tuned = cv_decorator(svm_tuned_precision)

        optimal_svm_pars, _, _ = optunity.maximize_structured(
            svm_tuned, _SVM_SEARCH_SPACE, num_evals=self._config.get('num_evals', 100)
        )

        self._model = _train_model(features_pca, self._labels, **optimal_svm_pars)

    def predict(self, flattened_query_image):
        """
        Predict the label of the given flattened image

        :param flattened_query_image: a flattened vector of the image
        :return: the predicted label
        """
        if len(flattened_query_image) != self._images_height * self._images_width:
            raise ValueError("Invalid length of flattened_query_image")

        query_image_transformed = self._pca.transform([flattened_query_image])

        return self._model.predict(query_image_transformed)[0]
