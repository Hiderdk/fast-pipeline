from abc import abstractmethod
from typing import Optional, Union
import polars as pl

from fast_pipeline.data_models import ColumnNames


class BaseLagGenerator():

    def __init__(self,
                 granularity: list[str],
                 features: Optional[list[str]],
                 iterations: list[int],
                 prefix: str,
                 grouping: Optional[list[str]],
                 are_estimator_features: bool = True,
                 ):

        self.features = features
        self.iterations = iterations
        self._features_out = []
        self._are_estimator_features = are_estimator_features
        self.granularity = granularity
        self.grouping = grouping
        if isinstance(self.granularity, str):
            self.granularity = [self.granularity]
        self.prefix = prefix
        self.column_names: Optional[ColumnNames] = None
        self._df = None

    @abstractmethod
    def generate_historical(self, df: pl.DataFrame, column_names: ColumnNames) -> pl.DataFrame:
        pass

    @abstractmethod
    def generate_future(self, df: pl.DataFrame) -> pl.DataFrame:
        pass

    @property
    def estimator_features_out(self) -> list[str]:
        if self._are_estimator_features:
            return self.features_out
        return []

    @property
    def features_out(self) -> list[str]:
        return self._features_out

    def _concat_df(self, df: pl.DataFrame) -> pl.DataFrame:

        df = df.with_columns(
            [pl.col(feature).cast(pl.Float64).alias(feature) for feature in self.features if feature in df.columns]
        )
        agg_dict = [
            pl.col(feature_name).mean().alias(feature_name)
            for feature_name in self.features
        ]

        if self.grouping:
            agg_dict_grouping = agg_dict.copy()
            agg_dict_grouping.append(pl.col(self.column_names.row_id).first())
            agg_dict_grouping.append(pl.col(self.column_names.sort).first())
            grouping_cols = self.grouping.copy() + self.granularity
            grouped_df = df.group_by(grouping_cols).agg(agg_dict_grouping)
        else:
            grouped_df = df

        agg_dict.append(pl.col(self.column_names.sort).first().alias(self.column_names.sort))
        granularity_grouped = grouped_df.group_by([self.granularity]).agg(agg_dict)

        concat_df = pl.concat([self._df, granularity_grouped], how="diagonal_relaxed")

        return concat_df.unique(
            subset=[self.column_names.row_id]
        )

    def _store_df(self, df: pl.DataFrame, additional_cols_to_use: Optional[list[str]] = None):

        df = df.with_columns(
            [pl.col(feature).cast(pl.Float64).alias(feature) for feature in self.features if feature in df.columns]
        )

        cols = list({
            *self.features,
            *self.granularity,
            self.column_names.row_id,
            self.column_names.sort,
        })

        if additional_cols_to_use:
            cols += additional_cols_to_use

        if self._df is None:
            self._df = df.select(cols)
        else:
            self._df = pl.concat([self._df, df.select(cols)])

        self._df = (self._df
                    .sort([self.column_names.sort],
                          )
                    .unique(subset=[self.column_names.row_id],
                            )
                    )

    def _create_transformed_df(self, df: pl.DataFrame, concat_df: pl.DataFrame) -> pl.DataFrame:

        ori_cols = [c for c in df.columns if c not in concat_df.columns] + [self.column_names.row_id]
        transformed_df = concat_df.join(df.select(ori_cols), on=[self.column_names.row_id], how='inner')
        return transformed_df.select(list(set(df.columns + self.features_out)))

    def reset(self) -> "BaseLagGenerator":
        self._df = None
        return self


class RollingMeanTransformer(BaseLagGenerator):

    def __init__(self,
                 window: int,
                 granularity: Union[list[str], str],
                 grouping: Optional[list[str]] = None,
                 features: Optional[list[str]] = None,
                 min_periods: int = 1,
                 are_estimator_features=True,
                 future_row_count: int = 10000,
                 prefix: str = 'rolling_mean_'):

        super().__init__(features=features, iterations=[window],
                         prefix=prefix, granularity=granularity, are_estimator_features=are_estimator_features,
                         grouping=grouping)
        self.window = window
        self.min_periods = min_periods
        self.future_row_count = future_row_count

    def generate_historical(self, df: pl.DataFrame, column_names: ColumnNames) -> pl.DataFrame:

        self.column_names = column_names
        if not self.features:
            self.features = [self.column_names.target]

        for feature_name in self.features:
            for lag in [self.window]:
                self._features_out.append(f'{self.prefix}{lag}_{feature_name}')

        if df.unique(subset=[column_names.row_id]).shape[0] != df.shape[
            0]:
            raise ValueError(
                f"Duplicated rows in df. Please make sure that the df has unique rows for {self.column_names.row_id}")

        self.column_names = column_names
        self._store_df(df)
        concat_df = self._concat_df(df)
        concat_df = self._generate_concat_df_with_feats(concat_df)
        transformed_df = self._create_transformed_df(df=df, concat_df=concat_df)
        cn = self.column_names
        recasts_mapping = {}
        for c in [cn.row_id]:
            if transformed_df[c].dtype != df[c].dtype:
                recasts_mapping[c] = df[c].dtype
        transformed_df = transformed_df.cast(recasts_mapping)
        df = df.join(transformed_df.select(cn.row_id, *self.features_out),
                     on=[cn.row_id], how='left')

        return df

    def generate_future(self, df: pl.DataFrame) -> pl.DataFrame:

        concat_df = pl.concat([self._df, df.select(self._df.columns)], how="diagonal_relaxed")
        concat_df = concat_df.tail(self.future_row_count)
        concat_df = self._generate_concat_df_with_feats(concat_df)
        future_row_ids = df.select(pl.col(self.column_names.row_id).cast(pl.Utf8).unique()).to_series()
        transformed_df = concat_df.filter(pl.col(self.column_names.row_id).cast(pl.Utf8).is_in(future_row_ids))
        transformed_df[self.features] = transformed_df[self.features].fill_nan(-999.21345)

        cn = self.column_names
        df = df.join(transformed_df.select(cn.row_id, *self.features_out),
                     on=[cn.row_id], how='left')

        return df

    def _generate_concat_df_with_feats(self, concat_df: pl.DataFrame) -> pl.DataFrame:

        if self._df is None:
            raise ValueError("fit_transform needs to be called before transform")

        rolling_means = [
            pl.col(feature_name)
            .shift()
            .rolling_mean(window_size=self.window, min_periods=self.min_periods)
            .over(self.granularity)
            .alias(f'{self.prefix}{self.window}_{feature_name}')
            for feature_name in self.features
        ]
        concat_df = concat_df.with_columns(rolling_means)

        concat_df = concat_df.sort(
            [self.column_names.sort])

        feats_added = [f for f in self.features_out if f in concat_df.columns]

        concat_df = concat_df.with_columns(
            [pl.col(f).forward_fill().over(self.granularity).alias(f) for f in feats_added]
        )
        return concat_df

    @property
    def features_out(self) -> list[str]:
        return self._features_out
