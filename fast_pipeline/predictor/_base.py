from abc import ABC, abstractmethod

import pandas as pd
import polars as pl


class Predictor(ABC):

    def __init__(self, estimator_features: list[str], target: str):
        self._estimator_features = estimator_features
        self.target = target
        self._pre_transformers: list[PredictorTransformer] = []

    @abstractmethod
    def train(self, df: pl.DataFrame) -> None:
        pass

    @abstractmethod
    def predict(self, df: pl.DataFrame) -> pl.DataFrame:
        pass

    def _fit_pre_transformers(self, df: pd.DataFrame) -> pd.DataFrame:
        for transformer in self._pre_transformers:
            df = transformer.fit_transform(df)
        return df

    def _transform_pre_transformers(self, df: pd.DataFrame) -> pd.DataFrame:
        for transformer in self._pre_transformers:
            df = transformer.transform(df)
        return df


class PredictorTransformer(ABC):

    @abstractmethod
    def fit(self, df: pd.DataFrame) -> None:
        pass

    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self.fit(df)
        return self.transform(df)
