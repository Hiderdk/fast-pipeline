import numpy as np
import polars as pl
from typing import Optional

from fast_pipeline.predictor._base import Predictor


class SkLearnPredictor(Predictor):

    def __init__(self, estimator,
                 estimator_features: list[str],
                 target: str,
                 ):
        self.estimator = estimator
        super().__init__(estimator_features=estimator_features, target=target)

    def train(self, df: pl.DataFrame) -> None:
        pandas_df = df.to_pandas()
        self.estimator.fit(pandas_df[self._estimator_features], pandas_df['target'])

    def predict(self, df: pl.DataFrame) -> np.ndarray:
        if hasattr(self.estimator, "predict_proba"):
            return self.estimator.predict_proba(df.select(self._estimator_features).to_pandas())
        else:
            return self.estimator.predict(df.select(self._estimator_features).to_pandas())
