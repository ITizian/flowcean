from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, cast, override

import polars as pl

from flowcean.core.model import Model

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray


class SupportsPredict(Protocol):
    """Protocol describing an object that has a `predict` method."""

    def predict(self, X: Any) -> NDArray: ...  # noqa: N803


class SupportsPredictProba(Protocol):
    """Protocol describing an object that has a `predict_proba` method."""

    def predict_proba(self, X: Any) -> NDArray: ...  # noqa: N803


class SciKitModel(Model):
    """A model that wraps a scikit-learn estimator."""

    estimator: SupportsPredict

    def __init__(
        self,
        estimator: SupportsPredict,
        *,
        input_features: Sequence[str],
        output_features: Sequence[str],
        name: str | None = None,
    ) -> None:
        """Initialize the model.

        Args:
            estimator: The scikit-learn estimator.
            input_features: The names of the input features.
            output_features: The names of the output features.
            name: The name of the model.
        """
        super().__init__()
        if name is None:
            name = estimator.__class__.__name__
        self._name = name
        self.estimator = estimator
        self.input_features = list(input_features)
        self.output_features = list(output_features)

    @override
    def _predict(
        self,
        input_features: pl.DataFrame | pl.LazyFrame,
    ) -> pl.LazyFrame:
        if isinstance(input_features, pl.LazyFrame):
            input_features = input_features.collect()

        outputs = self.estimator.predict(input_features)
        if len(self.output_features) == 1:
            data = {self.output_features[0]: outputs}
        else:
            data = {
                self.output_features[i]: outputs[:, i]
                for i in range(len(self.output_features))
            }
        return pl.LazyFrame(data)


class SciKitClassifierModel(SciKitModel):
    """A SciKit model for classifiers with probability predictions.

    Supports threshold-based predictions via the ``threshold`` attribute and
    exposes class probabilities via ``predict_proba``. The estimator must
    implement ``predict_proba``.
    """

    threshold: float

    def __init__(
        self,
        estimator: SupportsPredict,
        *,
        output_names: Iterable[str],
        threshold: float = 0.5,
        name: str | None = None,
    ) -> None:
        """Initialize the classifier model.

        Args:
            estimator: The scikit-learn classifier (must support
                ``predict_proba``).
            output_names: The names of the output columns.
            threshold: Decision threshold for the positive class
                (default: 0.5).
            name: The name of the model.
        """
        super().__init__(estimator, output_names=output_names, name=name)
        self.threshold = threshold

    def _predict_proba(
        self,
        input_features: pl.DataFrame | pl.LazyFrame,
    ) -> pl.LazyFrame:
        """Predict class probabilities (without preprocessing)."""
        if isinstance(input_features, pl.LazyFrame):
            input_features = input_features.collect()

        estimator = cast("SupportsPredictProba", self.estimator)
        probas = estimator.predict_proba(input_features)[:, 1]

        if len(self.output_names) == 1:
            data = {self.output_names[0]: probas}
        else:
            data = {
                self.output_names[i]: probas[:, i]
                for i in range(len(self.output_names))
            }
        return pl.LazyFrame(data)

    def predict_proba(
        self,
        input_features: pl.DataFrame | pl.LazyFrame,
    ) -> pl.LazyFrame:
        """Predict class probabilities, applying preprocessing transforms.

        Args:
            input_features: The inputs for which to predict probabilities.

        Returns:
            The predicted probabilities for the positive class.
        """
        input_features = self.preprocess(input_features)
        return self._predict_proba(input_features)

    @override
    def _predict(
        self,
        input_features: pl.DataFrame | pl.LazyFrame,
    ) -> pl.LazyFrame:
        if isinstance(input_features, pl.LazyFrame):
            input_features = input_features.collect()

        probas = self._predict_proba(input_features).collect()
        predictions = {}
        for col in probas.columns:
            predictions[col] = (probas[col] >= self.threshold).cast(pl.Int64)
        return pl.LazyFrame(predictions)
