# from .base import BaseEnsemble
class BaseEnsemble(BaseEstimator, MetaEstimatorMixin, metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, base_estimator, n_estimators=10,
                 estimator_params=tuple()):
        # Set parameters
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.estimator_params = estimator_params
        
class BaseWeightBoosting(BaseEnsemble, metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, 
                base_estimator=None,
                n_estimators=50,
                estimator_params=tuple(),
                learning_rate=1.,
                random_state=None):

        super().__init__(
            base_estimator=base_estimator,
            n_estimators=n_estimators,
            estimator_params=estimator_params)

        self.learning_rate = learning_rate
        self.random_state = random_state

class AdaBoost(BaseWeightBoosting, ClassifierMixin):
    def __init__(self,
                 base_estimator=None,
                 n_estimators=50,
                 learning_rate=1.,
                 algorithm='SAMME.R',
                 random_state=None):

        super().__init__(
            base_estimator=base_estimator,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            random_state=random_state)

        self.algorithm = algorithm

    def fit(self, X, y, sample_weight=None):
        # Check that algorithm is supported
        if self.algorithm not in ('SAMME', 'SAMME.R'):
            raise ValueError("algorithm %s is not supported" % self.algorithm)

        # Fit
        return super().fit(X, y, sample_weight)