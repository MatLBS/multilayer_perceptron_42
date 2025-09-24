import pandas as pd
from toolkit_mlp.math_utils import (calculate_count, calculate_mean,
                                    calculate_std, find_min, find_max,
                                    calculate_percentile, calculate_range)


def describe_column(column: pd.Series) -> pd.Series:
    """
    Describe a pandas Series by calculating various statistics.
    """
    column_described = pd.Series()
    column = column.dropna()

    count = calculate_count(column)
    mean = calculate_mean(column)
    var, std = calculate_std(column, mean)
    min = find_min(column)
    percentile_25 = calculate_percentile(column, 25)
    percentile_50 = calculate_percentile(column, 50)
    percentile_75 = calculate_percentile(column, 75)
    max = find_max(column)
    range = calculate_range(column)
    column_described["count"] = count
    column_described["mean"] = mean
    column_described["std"] = std
    column_described["min"] = min
    column_described["25%"] = percentile_25
    column_described["50%"] = percentile_50
    column_described["75%"] = percentile_75
    column_described["max"] = max
    column_described["range"] = range

    return column_described


def split_columns(feat_type: str, df: pd.DataFrame,
                  featured_list: list) -> None:
    """
    Splits the dataframe columns based on the feature
    type and calculates statistics for each feature.
    """
    cols = pd.DataFrame()
    match feat_type:
        case "Mean":
            cols = df.iloc[:, :10]
        case "Standard Error":
            cols = df.iloc[:, 10:20]
        case "Largest":
            cols = df.iloc[:, 20:]

    df_described = pd.DataFrame()
    cols.columns = [name for name in featured_list]
    for i in cols.columns:
        df_described[i] = describe_column(cols[i])
    print(f"--- FEATURE TYPE: {feat_type} ---")
    print(df_described)
    print()


def describe(file: str) -> pd.DataFrame:
    """
    Prints the summary statistics of a DataFrame.
    """

    df = pd.read_csv(file, header=None)
    df = df.drop(columns=[0, 1], axis=1)

    feature_list = [
        "Radius",
        "Texture",
        "Perimeter",
        "Area",
        "Smoothness",
        "Compactness",
        "Concavity",
        "Concave Points",
        "Symmetry",
        "Fractal Dimension",
    ]
    feat_type = ["Mean", "Standard Error", "Largest"]

    for i in feat_type:
        split_columns(i, df, feature_list)
