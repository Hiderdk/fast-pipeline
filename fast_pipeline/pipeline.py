import numpy as np
import polars as pl
from typing import Optional

from player_performance_ratings.predictor import BasePredictor

from fast_pipeline.data_models import ColumnNames
from fast_pipeline.transformer._base import BaseLagGenerator


class PipelinePredictor():

    def __init__(self,
                 lag_generators: list[BaseLagGenerator],
                 predictor: BasePredictor,
                 column_names: ColumnNames,
                 estimator_features: Optional[list[str]] = None,
                 ):
        self.estimator_features = estimator_features or []
        self.lag_generators = lag_generators
        self.predictor = predictor
        self.column_names = column_names

    def train(self, df: pl.DataFrame):
        estimator_features = self.estimator_features
        for lag_generator in self.lag_generators:
            df = lag_generator.generate_historical(df=df, column_names=self.column_names)
            estimator_features += lag_generator.estimator_features_out
        pandas_df = df.to_pandas()
        self.estimator.fit(pandas_df[estimator_features], pandas_df[self.column_names.target])
        self.estimator_features = estimator_features

    def predict(self, df: pl.DataFrame) -> np.ndarray:
        for lag_generator in self.lag_generators:
            df = lag_generator.generate_future(df)

        return self.predictor.predict(df.select(self.estimator_features).to_pandas())
