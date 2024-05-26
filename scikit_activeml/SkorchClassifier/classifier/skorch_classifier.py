import warnings

import numpy as np
import torch

from skactiveml.base import SkactivemlClassifier
from skorch import NeuralNet
from skactiveml.utils import is_labeled, MISSING_LABEL
from sklearn.utils.validation import check_is_fitted


class SkorchClassifier(NeuralNet, SkactivemlClassifier):
    """SkorchClassifier

    Implement a wrapper class, to make it possible to use `PyTorch` with
    `skactiveml`. This is achieved by providing a wrapper around `PyTorch`
    that has a skactiveml interface and also be able to handle missing labels.
    This wrapper is based on the open-source library `skorch` [1].

    Parameters
    ----------
    module : torch module (class or instance)
      A PyTorch :class:`~torch.nn.Module`. In general, the
      uninstantiated class should be passed, although instantiated
      modules will also work.
    criterion : torch criterion (class)
      The uninitialized criterion (loss) used to optimize the
      module.
    *args: arguments
        more possible arguments for initialize your neural network
        see: https://skorch.readthedocs.io/en/stable/net.html
    classes : array-like of shape (n_classes,), default=None
        Holds the label for each class. If none, the classes are determined
        during the fit.
    missing_label : scalar or string or np.nan or None, default=np.nan
        Value to represent a missing label.
    cost_matrix : array-like of shape (n_classes, n_classes)
        Cost matrix with `cost_matrix[i,j]` indicating cost of predicting class
        `classes[j]` for a sample of class `classes[i]`. Can be only set, if
        `classes` is not none.
    random_state : int or RandomState instance or None, default=None
        Determines random number for 'predict' method. Pass an int for
        reproducible results across multiple method calls.
    **kwargs : keyword arguments
        more possible parameters to customizing your neural network
        see: https://skorch.readthedocs.io/en/stable/net.html

    References
    ----------
    [1] Marian Tietz, Thomas J. Fan, Daniel Nouri, Benjamin Bossan, and
    skorch Developers. skorch: A scikit-learn compatible neural network
    library that wraps PyTorch, July 2017.
    """

    def __init__(
        self,
        module,
        criterion,
        classes=None,
        missing_label=MISSING_LABEL,
        cost_matrix=None,
        random_state=None,
        **kwargs,
    ):
        super(SkorchClassifier, self).__init__(
            module,
            criterion,
            **kwargs,
        )

        SkactivemlClassifier.__init__(
            self,
            classes=classes,
            missing_label=missing_label,
            cost_matrix=cost_matrix,
            random_state=random_state,
        )

        # set random state in PyTorch
        if isinstance(self.random_state, int):
            torch.manual_seed(self.random_state)

        # In Skorch, we don't need to initialize in the init statement, because
        # it will be called inside the fit function. But I think for the test I need
        # to call this here.
        # self.initialize()

    def fit(self, X, y, **fit_params):
        # check input parameters
        self.check_X_dict_ = {
            "ensure_min_samples": 0,
            "ensure_min_features": 0,
            "allow_nd": True,
            "dtype": None,
        }
        X, y, sample_weight = self._validate_data(
            X=X,
            y=y,
            check_X_dict=self.check_X_dict_,
        )

        is_lbld = is_labeled(y, missing_label=self.missing_label)

        if np.sum(is_lbld) == 0:
            raise ValueError("There is no labeled data.")
        else:
            X_lbld = X[is_lbld]
            y_lbld = y[is_lbld].astype(np.int64)
            return super(SkorchClassifier, self).fit(
                X_lbld, y_lbld, **fit_params
            )

    def predict(self, X):
        """Return class label predictions for the input data X.

        Parameters
        ----------
        X :  array-like, shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        y :  array-like, shape (n_samples)
            Predicted class labels of the input samples.
        """
        return SkactivemlClassifier.predict(self, X)

    def score(self, X, y, sample_weight=None):
        return SkactivemlClassifier.score(self, X, y)
