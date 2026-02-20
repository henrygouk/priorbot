from typing import Any, List
import numpy as np
from scipy.stats import dirichlet
from sklearn.base import BaseEstimator, clone
from .data import Dataset
from .priors import Prior, EmpiricalPrior

class DPGBClassifier:
    """
    Dirichlet Process Generalised Bayes classifier.

    This class provides a wrapper around conventional non-Bayesian methods by leveraging the ideas of predictive Bayes
    and generalised Bayes to create a flexible and efficient inference algorithm. It uses the predictive Bayes method
    of modelling uncertainty in observation space by placing a Dirichet Process prior over the possible data measures,
    then obtaining a posterior over data measures via conjugacy with the empirical distribution function. Generalised
    Bayes then tells us that we can sample a measure from this posterior and fit a model by minimising a loss function.
    Each of these point estimated models ends up being a sample from a generalised posterior. When the loss is a
    negative log likelihood, this is equivalent to sampling from the predictive Bayes posterior over classifiers.
    """

    def __init__(
            self,
            base_estimator: BaseEstimator,
            n_estimators: int = 100,
            alpha: float = 1,
            n_breaks: int = 1000,
            oversampling_factor: int = 10,
            random_state: int | None = None
        ):
        """
        Initialize the DPGBClassifier with a base estimator, base measure, and alpha.

        Parameters
        ----------
        base_estimator : object
            The base estimator to be used for prior estimation. Must support sample weighting.
        n_estimators : int, optional
            Number of estimators to use when approximating the posterior distribution (default is 100).
        alpha : float, optional
            Concentration parameter for the Dirichlet Process prior (default is 1.0).
        n_breaks : int, optional
            Number of breaks for the stick-breaking process (default is 1000).
        random_state : int, optional
            Random seed for reproducibility (default is None).
        """
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.alpha = alpha
        self.n_breaks = n_breaks
        self.oversampling_factor = oversampling_factor
        self.random_state = random_state

    def _fit_empirical_prior(self, dataset: Dataset, prior: EmpiricalPrior, **fit_params):
        """
        Fit the posterior in the case where the prior is a mixture of delta distributions.

        Parameters
        ----------
        dataset : Dataset
            The dataset to fit the model on, annotated with relevant meta-data.
        prior : EmpiricalPrior
            The prior distribution, which should be an EmpiricalPrior.
        fit_params : dict
            Additional parameters to be passed to the base estimator when fitting.
        """
        self.feature_names_ = dataset.feature_schema.get("properties", {}).keys()
        self.target_name_ = list(dataset.target_schema["properties"])[0]
        X = np.array([[data_point[feature] for feature in self.feature_names_] for data_point in dataset.data])
        y = np.array([data_point[self.target_name_] for data_point in dataset.data])

        if isinstance(prior, EmpiricalPrior) and self.alpha > 0:
            n_prior = len(prior.samples)
        else:
            n_prior = 0

        n_total = X.shape[0] + n_prior
        alpha = np.ones(n_total)

        if n_prior > 0:
            alpha[:n_prior] = self.alpha / n_prior
            prior_X = np.array([[sample[feature] for feature in self.feature_names_] for sample in prior.samples])
            prior_y = np.array([sample[self.target_name_] for sample in prior.samples])
            X = np.vstack([prior_X, X])
            y = np.hstack([prior_y, y])

        dirichlet_samples = dirichlet.rvs(alpha, size=self.n_estimators, random_state=self.random_state)

        self.estimators_ = [
            clone(self.base_estimator) for _ in range(self.n_estimators)
        ]

        for i in range(self.n_estimators):
            self.estimators_[i].fit(X, y, sample_weight=dirichlet_samples[i], **fit_params)

        return self

    def _fit_stick_breaking(self, dataset: Dataset, prior: Prior, **fit_params):
        """
        Fit the posterior in the case where the prior is a stick-breaking process.

        Parameters
        ----------
        dataset : Dataset
            The dataset to fit the model on, annotated with relevant meta-data.
        prior : Prior
            The prior distribution, which should be a stick-breaking process.
        fit_params : dict
            Additional parameters to be passed to the base estimator when fitting.
        """
        self.estimators_ = [
            self.base_estimator.__class__(**self.base_estimator.get_params())
            for _ in range(self.n_estimators)
        ]
        self.feature_names_ = dataset.feature_schema.get("properties", {}).keys()
        self.target_name_ = list(dataset.target_schema["properties"])[0]
        X = np.array([[data_point[feature] for feature in self.feature_names_] for data_point in dataset.data])
        y = np.array([data_point[self.target_name_] for data_point in dataset.data])

        joint_schema = dataset.feature_schema.copy()
        joint_schema["properties"].update(dataset.target_schema["properties"])

        for i in range(self.n_estimators):
            beta = np.random.beta(1, self.alpha + X.shape[0], size=self.n_breaks)
            weights = np.cumprod(np.concatenate(([1], 1 - beta[:-1]))) * beta
            weights /= np.sum(weights)

            # Shuffle the weights to ensure randomness
            np.random.seed(self.random_state)
            np.random.shuffle(weights)

            # Determine number of samples from base measure and number of samples from X
            n_prior = int(self.n_breaks * self.alpha / (self.alpha + X.shape[0]))
            n_samples = self.n_breaks - n_prior

            prior_data = prior.sample(n_prior, joint_schema)
            prior_X = np.array([[data_point[feature] for feature in self.feature_names_] for data_point in prior_data])
            prior_y = np.array([data_point[self.target_name_] for data_point in prior_data])

            real_indices = np.random.choice(X.shape[0], n_samples, replace=True)
            (real_X, real_y) = (X[real_indices], y[real_indices])
            
            combined_X = np.vstack([prior_X, real_X])
            combined_y = np.hstack([prior_y, real_y])
            
            self.estimators_[i].fit(combined_X, combined_y, sample_weight=weights, **fit_params)

        return self

    def fit_informative(self, dataset: Dataset, prior: Prior, **fit_params):
        """
        Fit the model to the data while using an informative prior.

        Parameters
        ----------
        dataset : Dataset
            The dataset to fit the model on, annotated with relevant meta-data.
        prior : Prior
            The prior distribution, which can be either an EmpiricalPrior or a stick-breaking process.
        fit_params : dict
            Additional parameters to be passed to the base estimator when fitting.
        """
        if isinstance(prior, EmpiricalPrior):
            return self._fit_empirical_prior(dataset, prior, **fit_params)
        else:
            return self._fit_stick_breaking(dataset, prior, **fit_params)
        
    def predict_proba_dict(self, X: List[dict[str, Any]]):
        """
        Predict class probabilities for the input data.

        Parameters
        ----------
        X : dict[str, Any]
            Input data for which to predict class probabilities.

        Returns
        -------
        array-like, shape (n_samples, n_classes)
            Predicted class probabilities.
        """
        if not hasattr(self, 'estimators_'):
            raise RuntimeError("The model has not been fitted yet.")

        X_sklearn = np.array([[data_point[feature] for feature in self.feature_names_] for data_point in X])

        # Aggregate predictions from all estimators
        predictions = np.array([estimator.predict_proba(X_sklearn) for estimator in self.estimators_])
        return np.mean(predictions, axis=0)
    
    def predict_dict(self, X: List[dict[str, Any]]):
        """
        Predict class labels for the input data.

        Parameters
        ----------
        X : dict[str, Any]
            Input data for which to predict class labels.

        Returns
        -------
        array-like, shape (n_samples,)
            Predicted class labels.
        """
        proba = self.predict_proba_dict(X)
        return np.argmax(proba, axis=1) if proba.ndim > 1 else np.round(proba).astype(int)

