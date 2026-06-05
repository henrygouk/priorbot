from typing import Any, List
import numpy as np
from scipy.stats import dirichlet
from sklearn.base import BaseEstimator, clone
from sklearn.pipeline import Pipeline
from .data import Dataset
from .priors import Prior, EmpiricalPrior

def roc_auc_score(y_true, y_proba_dicts):
    """
    Compute the ROC AUC score for a multi-class classification problem.

    Parameters
    ----------
    y_true : array-like, shape (n_samples,)
        True class labels.
    y_proba_dicts : list of dict
        List of dictionaries containing predicted probabilities for each class. Each dictionary should have class labels as keys and predicted probabilities as values.

    Returns
    -------
    float
        The computed ROC AUC score.
    """
    from sklearn.metrics import roc_auc_score as sklearn_roc_auc_score

    # Extract class labels and predicted probabilities
    classes = sorted(y_proba_dicts[0].keys())
    y_proba = np.array([
        [proba_dict[cls_name] for cls_name in classes] for proba_dict in y_proba_dicts
    ])

    if len(classes) == 2:
        y_proba = y_proba[:, 1]

    y_true_idx = np.zeros(len(y_true))

    for i, label in enumerate(y_true):
        y_true_idx[i] = classes.index(label)

    return sklearn_roc_auc_score(y_true_idx, y_proba, multi_class="ovr")

def log_likelihood(y_true, y_proba_dicts):
    """
    Compute the log likelihood of the true labels given the predicted probabilities.

    Parameters
    ----------
    y_true : array-like, shape (n_samples,)
        True class labels.
    y_proba_dicts : list of dict
        List of dictionaries containing predicted probabilities for each class. Each dictionary should have class labels as keys and predicted probabilities as values.

    Returns
    -------
    float
        The computed log likelihood.
    """
    log_likelihood = 0.0

    for i in range(len(y_true)):
        proba_dict = y_proba_dicts[i]
        true_label = y_true[i]

        if true_label in proba_dict:
            log_likelihood += np.log(max(proba_dict[true_label], 1e-8))
        else:
            log_likelihood += np.log(1e-8)

    return log_likelihood

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

    def _get_class_index(self, data_point: dict[str, Any]) -> int | None:
        if self.target_name_ in data_point:
            target_value = data_point[self.target_name_]
            return self.classes_.index(target_value)
        else:
            return None

    def _to_sklearn_format(self, data: List[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
        """
        Convert a list of dictionaries to the format expected by scikit-learn.

        Categorical features and targets will be encoded as integers based on their index in the enum list in the schema.

        Parameters
        ----------
        data : list of dict
            The input data, where each element is a dictionary representing a data point.

        Returns
        -------
        tuple of (X, y)
            X is a 2D numpy array of shape (n_samples, n_features) containing the feature values,
            and y is a 1D numpy array of shape (n_samples,) containing the target values.
        """
        if len(data) == 0:
            return np.empty((0, len(self.feature_names_))), np.empty((0,))

        X = np.array([[data_point[feature] for feature in self.feature_names_] for data_point in data])
        y = np.array([self._get_class_index(data_point) for data_point in data])

        # Convert categorical features and targets to integer indices based on the schema
        for idx, f in enumerate(self.feature_names_):
            if f in self.feature_maps_:
                feature_map = self.feature_maps_[f]
                X[:, idx] = np.array([feature_map[value] for value in X[:, idx]])

        return X.astype(float), y

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

        X, y = self._to_sklearn_format(dataset.data)

        if isinstance(prior, EmpiricalPrior) and self.alpha > 0:
            n_prior = len(prior.samples)
            prior_samples = self._to_sklearn_format(prior.samples)
        else:
            n_prior = 0

        n_total = len(X) + n_prior
        alpha = np.ones(n_total)

        if n_prior > 0:
            alpha[:n_prior] = self.alpha / n_prior
            # prior_X = np.array([[sample[feature] for feature in self.feature_names_] for sample in prior_samples])
            # prior_y = np.array([sample[self.target_name_] for sample in prior_samples])
            prior_X, prior_y = prior_samples

            if len(X) > 0:
                X = np.vstack([prior_X, X])
                y = np.hstack([prior_y, y])
            else:
                X = prior_X
                y = prior_y

        dirichlet_samples = dirichlet.rvs(alpha, size=self.n_estimators, random_state=self.random_state)

        self.estimators_ = [
            clone(self.base_estimator) for _ in range(self.n_estimators)
        ]

        for i in range(self.n_estimators):
            fit_kwargs = {"sample_weight": dirichlet_samples[i], **fit_params}

            if isinstance(self.estimators_[i], Pipeline):
                fit_kwargs = {f"{self.estimators_[i].steps[-1][0]}__sample_weight": dirichlet_samples[i], **fit_params}

            self.estimators_[i].fit(X, y, **fit_kwargs)

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

        X, y = self._to_sklearn_format(dataset.data)

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
            prior_X, prior_y = self._to_sklearn_format(prior_data)

            real_indices = np.random.choice(X.shape[0], n_samples, replace=True)
            (real_X, real_y) = (X[real_indices], y[real_indices])
            
            combined_X = np.vstack([prior_X, real_X])
            combined_y = np.hstack([prior_y, real_y])
            
            fit_kwargs = {"sample_weight": weights, **fit_params}

            if isinstance(self.estimators_[i], Pipeline):
                fit_kwargs = {f"{self.estimators_[i].steps[-1][0]}__sample_weight": weights, **fit_params}

            self.estimators_[i].fit(combined_X, combined_y, **fit_kwargs)

        return self

    def fit_informative(self, dataset: Dataset, prior: Prior, **fit_params):
        """
        Fit the model to the data while using an informative prior.

        Parameters
        ----------
        dataset : Dataset
            The dataset to fit the model on, annotated with relevant meta-data.
        prior : Prior
            The prior predictive distribution
        fit_params : dict
            Additional parameters to be passed to the base estimator when fitting.
        """
        self.feature_names_ = list(dataset.feature_schema.get("properties", {}).keys())
        self.feature_maps_ = {}
        self.target_name_ = list(dataset.target_schema["properties"])[0]
        
        for f in self.feature_names_:
            schema = dataset.feature_schema["properties"][f]

            if schema["type"] == "string":
                feature_map = {value: index for index, value in enumerate(schema["enum"])}
                self.feature_maps_[f] = feature_map

        target_schema = dataset.target_schema["properties"][self.target_name_]

        if target_schema["type"] != "string":
            raise ValueError("Target variable must be categorical (type 'string') for classification.")

        target_schema = dataset.target_schema["properties"][self.target_name_]
        self.classes_ = list(target_schema["enum"])

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

        X_sklearn, _ = self._to_sklearn_format(X)

        # Aggregate predictions from all estimators
        predictions = np.array([estimator.predict_proba(X_sklearn) for estimator in self.estimators_])
        proba = np.mean(predictions, axis=0)

        # Convert to dict format
        proba_dicts = []

        for i in range(proba.shape[0]):
            if len(self.classes_) == 2:
                proba_dict = {self.classes_[0]: proba[i, 0], self.classes_[1]: 1.0 - proba[i, 0]}
            else:
                proba_dict = {self.classes_[j]: proba[i, j] for j in range(proba.shape[1])}

            proba_dicts.append(proba_dict)

        return proba_dicts
    
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
        proba_dicts = self.predict_proba_dict(X)
        predictions = [max(proba_dict, key=proba_dict.get) for proba_dict in proba_dicts]
        return predictions

