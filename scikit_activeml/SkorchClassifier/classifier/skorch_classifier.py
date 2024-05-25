import numpy as np

from skactiveml.base import SkactivemlClassifier
from skorch import NeuralNet
from skactiveml.utils import is_labeled, MISSING_LABEL
from sklearn.utils.validation import check_is_fitted


class SkorchClassifier(NeuralNet, SkactivemlClassifier):
    def __init__(
            self,
            module,
            *args,
            classes=None,
            missing_label=MISSING_LABEL,
            cost_matrix=None,
            random_state=None,
            **module_kwargs,
    ):
        n_classes = len(classes)
        super(SkorchClassifier, self).__init__(
            module,
            *args,
            module__n_classes=n_classes,
            **module_kwargs,
        )

        SkactivemlClassifier.__init__(
            self,
            classes=classes,
            missing_label=missing_label,
            cost_matrix=cost_matrix,
            random_state=random_state,
        )

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

        # check whether model is a valid model

        is_lbld = is_labeled(y, missing_label=self.missing_label)
        try:
            X_lbld = X[is_lbld]
            y_lbld = y[is_lbld].astype(np.int64)
            super(SkorchClassifier, self).fit(X_lbld, y_lbld, **fit_params)
            self.is_fitted = True
            return self
        except Exception as e:
            self.is_fitted_ = False
            return self

    def initialize(self):
        super(SkorchClassifier, self).check_training_readiness()

        super(SkorchClassifier, self)._initialize_virtual_params()
        super(SkorchClassifier, self)._initialize_callbacks()
        super(SkorchClassifier, self)._initialize_module()
        super(SkorchClassifier, self)._initialize_criterion()
        super(SkorchClassifier, self)._initialize_optimizer()
        super(SkorchClassifier, self)._initialize_history()

        self.initialized_ = True
        return self

    def predict_proba(self, X, predict_nonlinearity:callable=None, **kwargs):
        # Alternative 1: pass the parameter ```predict_nonlinearity: callable``` by instance creation
        # original from Skorch, actually, in the instance predict_nonlinearity='auto',  When set to ‘auto’,
        # infers the correct nonlinearity based on the criterion
        # (softmax for CrossEntropyLoss and sigmoid for BCEWithLogitsLoss).
        # see: https://skorch.readthedocs.io/en/stable/classifier.html# (search: predict_nonlinearity)

        # Alternative 2: pass the ```predict_nonlinearity: callable``` in the predict_proba function and also the
        # corresponding arguments for this callable.
        return super(SkorchClassifier, self).predict_proba(X)


    def predict(self, X):
        check_is_fitted(self)
        if self.is_fitted:
            return SkactivemlClassifier.predict(self, X)
        else:
            p = self.predict_proba([X[0]])[0]
            y_pred = self.random_state_.choice(
                np.arange(len(self.classes_)), len(X), replace=True, p=p
            )
            return y_pred
