import os
import random
import numpy as np
import pandas as pd

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

def memory_usage_mb(df: pd.DataFrame) -> float:
    return df.memory_usage(deep=True).sum() / 1024**2

def basic_summary(df: pd.DataFrame) -> pd.DataFrame:
    s = df.describe(include="all", datetime_is_numeric=True).T
    s["missing"] = df.isna().sum()
    s["missing_pct"] = s["missing"] / len(df)
    s["dtype"] = df.dtypes.astype(str)
    return s

def reduce_categoricals(series: pd.Series, min_frac=0.01, other_label="Other"):
    freq = series.value_counts(normalize=True)
    keep = set(freq[freq >= min_frac].index)
    return series.where(series.isin(keep), other_label)
