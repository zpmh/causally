import numpy as np
from typing import Callable
from torch import nn, from_numpy
from abc import ABCMeta, abstractmethod
from sklearn.gaussian_process.kernels import PairwiseKernel
from sklearn.linear_model import LinearRegression


# Base class for causal mechanism generation
class PredictionModel(metaclass=ABCMeta):
    @abstractmethod
    def predict(self, X: np.array) -> np.array:
        raise NotImplementedError


# * Linear mechanisms *
class LinearMechanism(PredictionModel):
    """Linear causal mechanism by linear regression.

    Parameters
    ----------
    min_weight: float, default -1
        Minimum value of causal mechanisms weights
    max_weight: float, default 1
        Maximum value of causal mechanisms weights
    min_abs_weight: float, default 0.05
        Minimum value of the absolute value of any causal mechanism weight.
        Low value of min_abs_weight potentially lead to lambda-unfaithful distributions.
    """

    def __init__(
        self,
        min_weight: float = -1.0,
        max_weight: float = 1.0,
        min_abs_weight: float = 0.05,
    ):
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.min_abs_weight = min_abs_weight

        # One of max_weight or min_weight must be larger (abs value) than min_abs_weight
        if not (abs(max_weight) > min_abs_weight or abs(min_weight) > min_abs_weight):
            raise ValueError(
                "The range of admitted weights is empty. Please set"
                " one between ``min_weight`` and ``max_weight`` with absolute"
                "  value larger than ``min_abs_weight``."
            )

        self.linear_reg = LinearRegression(fit_intercept=False)

    def predict(self, X: np.array) -> np.array:
        """Transform X via a linear causal mechanism.

        Parameters
        ----------
        X: np.array of shape (num_samples, num_parents)
            Samples of the parents to be transformed by the causal mechanism.

        Returns
        -------
        y: np.array
            The output of the causal mechanism
        """
        if X.ndim != 2:
            X = X.reshape((-1, 1))
        n_covariates = X.shape[1]

        # Random initialization of the causal mechanism
        self.linear_reg.coef_ = np.random.uniform(
            self.min_weight, self.max_weight, n_covariates
        )

        # Reject ~0 coefficients
        for i in range(n_covariates):
            while abs(self.linear_reg.coef_[i]) < self.min_abs_weight:
                self.linear_reg.coef_[i] = np.random.uniform(
                    self.min_weight, self.max_weight, 1
                )

        self.linear_reg.intercept_ = 0
        effect = self.linear_reg.predict(X)
        return effect


# * Nonlinear mechanisms *
class NeuralNetMechanism(PredictionModel):
    """Nonlinear causal mechanism parametrized by a neural network.

    The transformation is parametrized by a simple neural network with
    one hidden layer, followed by an activation function, LayerNorm,
    and the linear output layer.

    Parameters
    ----------
    weights_mean: float, default 0
        Average value of the initialized weights.
    weights_std: float, default 0
        Standard deviation of the initialized weights.
    hidden_dim: int, default 10
        Number of neurons in the hidden layer
    activation: nn.Module, default ``nn.PReLU``
        The nonlinear activation function.
    """

    def __init__(
        self,
        weights_mean: float = 0.0,
        weights_std: float = 1.0,
        hidden_dim: int = 10,
        activation: nn.Module = nn.PReLU(),
    ):
        self.weights_mean = weights_mean
        self.weights_std = weights_std
        self.hidden_dim = hidden_dim
        self.activation = activation

        self._model = None

    def predict(self, X: np.array) -> np.array:
        """Generate the effect given the parents.

        The effect is generated as a nonlinear function parametrized by a neural network
        with a single hidden layer.

        Parameters
        ----------
        X: np.array of shape (num_samples, num_parents)
            Input of the neural network with the parent node instances.

        Returns
        -------
        effect: np.array of shape (num_samples)
            The output of the neural network with X as input.
        """
        if X.ndim != 2:
            X = X.reshape((-1, 1))
        n_samples = X.shape[0]
        n_causes = X.shape[1]

        # Make architecture
        self._model = nn.Sequential(
            nn.Linear(n_causes, self.hidden_dim),
            self.activation,
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, 1),
        )
        self._model.apply(self._weight_init)

        # Convert to torch.Tensor for forward pass
        data = X.astype("float32")
        data = from_numpy(data)

        # Forward pass
        effect = np.reshape(self.model(data).data, (n_samples,))

        return effect.numpy()

    def _weight_init(self, module):
        """Random initialization of model weights."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(
                module.weight.data, mean=self.weights_mean, std=self.weights_std
            )

    @property
    def model(self):
        """Return the nn.Module instance of the neural network architecture."""
        if self._model is None:
            raise ValueError(
                "Torch model not initialized. Call ``self.predict()`` first"
            )
        return self._model


class GaussianProcessMechanism(PredictionModel):
    """Nonlinear causal mechanism sampled from a gaussian process.

    The nonlinear transformation is generated sampling the effect from a
    gaussian process with covariance matrix defined as the kernel matrix of the
    parents' observations.

    Parameters
    ----------
    gamma: float, default 1
        The gamma parameters fixing the variance of the kernel.
        Larger values of gamma determines bigger magnitude of the causal mechanisms.
    """

    def __init__(self, gamma: float = 1.0):
        self.rbf = PairwiseKernel(gamma=gamma, metric="rbf")

    def predict(self, X: np.array) -> np.array:
        """Generate the effect given the parents.

        The effect is generated as a nonlinear function sampled from a
        gaussian process.

        Parameters
        ----------
        X: np.array of shape (num_samples, num_parents)
            Input of the RBF kernel.

        Returns
        -------
        effect: np.array of shape (num_samples)
            Causal effect sampled from the gaussian process with
            covariance matrix given by the RBF kernel with X as input.
        """
        num_samples = X.shape[0]
        # rbf = GPy.kern.RBF(input_dim=X.shape[1],lengthscale=self.lengthscale,variance=self.f_magn)
        # covariance_matrix = rbf.K(X,X)
        covariance_matrix = self.rbf(X, X)

        # Sample the effect as a zero centered normal with covariance given by the RBF kernel
        effect = np.random.multivariate_normal(np.zeros(num_samples), covariance_matrix)
        return effect


# Base class for PostnonLinearModel invertible functions
class InvertibleFunction:
    """Invertible functions for the post-nonlinear model abstract class.

    This class can be used to define the invertible transformation for the
    structural equations of a PostNonlinear model. In order to instantiate
    an InvertibleFunction, simply pass

    Parameters
    ----------
    function: Callable
        Python function implementing an invertible mathematical transformation.
    """

    def __init__(self, function: Callable) -> None:
        self.function = function

    def forward(self, input: np.array):
        """Apply the invertible function to the input."""
        return self.function(input)

    def __call__(self, input: np.array):
        return self.forward(input)
