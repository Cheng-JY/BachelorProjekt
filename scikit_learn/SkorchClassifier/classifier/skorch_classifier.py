from skactiveml.base import SkactivemlClassifier
from skorch import NeuralNet
from skactiveml.utils import is_labeled, MISSING_LABEL

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

        self.regularized = module_kwargs.get("regularized")
        self.lambda1 = module_kwargs.get("lambda1")

    def get_loss(self, y_pred, y_true, *args, **kwargs):
        loss = super(SkorchClassifier, self).get_loss(y_pred, y_true, *args, **kwargs)
        if self.regularized is not None:
            if self.regularized == 1 and self.lambda1 is float:
                loss += self.lambda1 * sum([w.abs().sum() for w in self.module_.parameters()])
        return loss

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
            return super(SkorchClassifier, self).fit(X_lbld, y_lbld, **fit_params)
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

    def predict(self, X):
        return SkactivemlClassifier.predict(self, X)