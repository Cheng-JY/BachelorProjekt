import warnings

import numpy as np
import torch

from skactiveml.base import SkactivemlClassifier
from skorch import NeuralNet
from skactiveml.utils import is_labeled, MISSING_LABEL
from sklearn.utils.validation import check_is_fitted


class SkorchClassifier(NeuralNet, SkactivemlClassifier):
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
        """Initialize and fit the module.

        If the module was already initialized, by calling fit, the
        module will be re-initialized (unless ``warm_start`` is True).

        Parameters
        ----------
        X : matrix-like, shape (n_samples, n_features)
            Training data set, usually complete, i.e. including the labeled and
            unlabeled samples
        y : array-like of shape (n_samples, )
            Labels of the training data set (possibly including unlabeled ones
            indicated by self.missing_label)
        fit_params : dict-like
            Further parameters as input to the 'fit' method of the 'estimator'.

        Returns
        -------
        self: SkorchClassifier,
            The SkorchClassifier is fitted on the training data.
        """

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
        try:
            if np.sum(is_lbld) == 0:
                raise ValueError("There is no labeled data.")
            else:
                X_lbld = X[is_lbld]
                y_lbld = y[is_lbld].astype(np.int64)
                return super(SkorchClassifier, self).fit(
                    X_lbld, y_lbld, **fit_params
                )
        except Exception as e:
            warnings.warn(
                "The 'base_estimator' could not be fitted because of"
                " '{}'. ".format(e)
            )
            return self

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

