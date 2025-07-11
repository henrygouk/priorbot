from abc import ABCMeta, abstractmethod
import numpy as np
from scipy.stats import dirichlet, multivariate_normal, uniform
from sklearn.base import BaseEstimator, MetaEstimatorMixin

class DiracMixture:
    def __init__(self, X, y=None):
        """
        Initialize a Dirac mixture with given data points.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Data points to initialize the Dirac mixture.
        y : array-like, shape (n_samples,), optional
            Target values associated with the data points (default is None).
        """
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        
        if y is not None and not isinstance(y, np.ndarray):
            y = np.array(y)
        
        if y is not None and X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples.")

        self.X = X
        self.y = y

    @classmethod
    def from_gaussian(cls, mu, sigma, n_samples, n_classes=None, random_state=None):
        """
        Create a Dirac mixture that approximates an iso Gaussian distribution.

        Parameters
        ----------
        mu : array-like, shape (n_features,)
            Mean of the Gaussian distribution.
        sigma : array-like, shape (n_features, n_features)
            Covariance matrix of the Gaussian distribution.
        n_samples : int
            Number of samples to draw from the Gaussian distribution.

        Returns
        -------
        DiracMixture
            A Dirac mixture initialized with samples from the Gaussian distribution.
        """
        X = multivariate_normal.rvs(mean=mu, cov=sigma, size=n_samples, random_state=random_state)

        if n_classes is not None and n_classes > 1:
            # Assign random class labels if multiple classes are specified
            y = uniform.rvs(0, n_classes, size=n_samples, random_state=random_state).astype(int)
        else:
            y = None

        return cls(X, y)
    
class Gaussian:
    def __init__(self, n_features, n_classes):
        self.n_features = n_features
        self.n_classes = n_classes

    def sample(self, n_samples, random_state=None):
        """        Sample from a Gaussian distribution.

        Parameters
        ----------
        n_samples : int
            Number of samples to draw.
        random_state : int, optional
            Random seed for reproducibility (default is None).

        Returns
        -------
        tuple
            A tuple containing the sampled features and their corresponding class labels.
        """
        rng = np.random.default_rng(random_state)
        X = rng.normal(size=(n_samples, self.n_features))
        y = rng.integers(0, self.n_classes, size=n_samples)
        return X, y

class BaseBayes(MetaEstimatorMixin, BaseEstimator, metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, base_estimator):
        """
        Initialize the BaseBayes with a base estimator.

        Parameters
        ----------
        base_estimator : object
            The base estimator to be used for prior estimation.
        """
        self.base_estimator = base_estimator

class GenBayesClassifier(BaseBayes):
    def __init__(self, base_estimator, base_measure, n_estimators=100, alpha=1, n_breaks=1000, random_state=None):
        """
        Initialize the GenBayesClassifier with a base estimator, base measure, and alpha.

        Parameters
        ----------
        base_estimator : object
            The base estimator to be used for prior estimation.
        base_measure : object
            The base measure for the Dirichlet Process.
        n_estimators : int, optional
            Number of estimators to use when approximating the posterior distribution (default is 50).
        alpha : float, optional
            Concentration parameter for the Dirichlet Process (default is 1.0).
        n_breaks : int, optional
            Number of breaks for the stick-breaking process (default is 1000).
        random_state : int, optional
            Random seed for reproducibility (default is None).
        """
        super().__init__(base_estimator)
        self.base_measure = base_measure
        self.n_estimators = n_estimators
        self.alpha = alpha
        self.n_breaks = n_breaks
        self.random_state = random_state

    def _fit_dirac_mixture(self, X, y, **fit_params):
        """
        Fit the posterior in the case where the prior is a Dirac mixture.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data.
        y : array-like, shape (n_samples,)
            Target values.
        fit_params : dict
            Additional parameters to be passed to the base estimator when fitting.
        """
        n_prior = self.base_measure.X.shape[0] if self.alpha > 0 else 0
        num = X.shape[0] + n_prior
        alpha = np.ones(num)

        if n_prior > 0:
            if self.base_measure.y is None:
                raise ValueError("Base measure must have associated labels (y) for Dirac mixture.")
            
            alpha[:n_prior] = self.alpha / n_prior
            X = np.vstack([self.base_measure.X, X])
            y = np.hstack([self.base_measure.y, y])

        dirichlet_samples = dirichlet.rvs(alpha, size=self.n_estimators, random_state=self.random_state)

        self.estimators_ = [
            self.base_estimator.__class__(**self.base_estimator.get_params())
            for _ in range(self.n_estimators)
        ]

        for i in range(self.n_estimators):
            self.estimators_[i].fit(X, y, sample_weight=dirichlet_samples[i], **fit_params)

        return self

    def _fit_stick_breaking(self, X, y, **fit_params):
        """
        Fit the posterior in the case where the prior is a stick-breaking process.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data.
        y : array-like, shape (n_samples,)
            Target values.
        fit_params : dict
            Additional parameters to be passed to the base estimator when fitting.
        """
        self.estimators_ = [
            self.base_estimator.__class__(**self.base_estimator.get_params())
            for _ in range(self.n_estimators)
        ]

        n_breaks = self.n_breaks if self.n_breaks is not None else X.shape[0]

        for i in range(self.n_estimators):
            beta = np.random.beta(1, self.alpha + X.shape[0], size=n_breaks)
            weights = np.cumprod(np.concatenate(([1], 1 - beta[:-1]))) * beta
            weights /= np.sum(weights)

            # Shuffle the weights to ensure randomness
            np.random.seed(self.random_state)
            np.random.shuffle(weights)

            # Determine number of samples from base measure and number of samples from X
            n_prior = int(n_breaks * self.alpha / (self.alpha + X.shape[0]))
            n_samples = n_breaks - n_prior

            (prior_X, prior_y) = self.base_measure.sample(n_prior, random_state=self.random_state)
            real_indices = np.random.choice(X.shape[0], n_samples, replace=True)
            (real_X, real_y) = (X[real_indices], y[real_indices])
            combined_X = np.vstack([prior_X, real_X])
            combined_y = np.hstack([prior_y, real_y])
            self.estimators_[i].fit(combined_X, combined_y, sample_weight=weights, **fit_params)

        return self

    def fit(self, X, y, **fit_params):
        """
        Fit the model to the data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data.
        y : array-like, shape (n_samples,)
            Target values.
        fit_params : dict
            Additional parameters to be passed to the base estimator when fitting.
        """
        if isinstance(self.base_measure, DiracMixture) or self.alpha == 0:
            return self._fit_dirac_mixture(X, y, **fit_params)
        else:
            return self._fit_stick_breaking(X, y, **fit_params)
        
    def predict_proba(self, X):
        """
        Predict class probabilities for the input data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input data for which to predict class probabilities.

        Returns
        -------
        array-like, shape (n_samples, n_classes)
            Predicted class probabilities.
        """
        if not hasattr(self, 'estimators_'):
            raise RuntimeError("The model has not been fitted yet.")
        
        # Aggregate predictions from all estimators
        predictions = np.array([estimator.predict_proba(X) for estimator in self.estimators_])
        return np.mean(predictions, axis=0)
    
    def predict(self, X):
        """
        Predict class labels for the input data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input data for which to predict class labels.

        Returns
        -------
        array-like, shape (n_samples,)
            Predicted class labels.
        """
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1) if proba.ndim > 1 else np.round(proba).astype(int)

# Test the implementation using a simple example: a two class problem where each class is a 2D Gaussian distribution
if __name__ == "__main__":
    # Example usage
    from sklearn.datasets import make_classification
    from sklearn.linear_model import LogisticRegression

    # Create a synthetic dataset
    X, y = make_classification(n_samples=50, n_features=2, n_informative=2, n_redundant=0, random_state=42)

    # Initialize a base measure as a Dirac mixture
    base_measure = Gaussian(2, 2)
    #base_measure = DiracMixture.from_gaussian(mu=np.zeros(2), sigma=np.eye(2), n_samples=200, n_classes=2, random_state=42)

    # Initialize the base estimator
    base_estimator = LogisticRegression(penalty=None)

    # Initialize the GenBayesClassifier
    model = GenBayesClassifier(base_estimator=base_estimator, base_measure=base_measure, n_estimators=1000, alpha=10, random_state=41)

    # Fit the model
    model.fit(X, y)

    # Predict class labels
    predictions = model.predict_proba(X)

    # Print the AUC
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(y, predictions[:, 1])
    print(f"AUC: {auc:.4f}")

    # Print the negative log likelihood
    from sklearn.metrics import log_loss
    nll = log_loss(y, predictions)
    print(f"Negative Log Likelihood: {nll:.4f}")