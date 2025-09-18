import pandas as pd
import numpy as np


def calculate_count(column: pd.Series) -> int:
    """
    Calculate the count of non-null values in a pandas Series.
    """
    sum = 0
    for i in column:
        sum += 1
    return sum


def calculate_mean(column: pd.Series) -> float:
    """
    Calculate the mean of a pandas Series.
    """
    sum = 0
    for i in column:
        sum += i
    return sum / len(column)


def calculate_std(column: pd.Series, mean) -> float:
    """
    Calculate the standard deviation of a pandas Series.
    """
    sum = 0
    for i in column:
        sum += (i - mean) ** 2
    variance = sum / (len(column) - 1)
    return variance, variance ** 0.5


def find_min(column: pd.Series) -> float:
    """
    Find the minimum value in a pandas Series.
    """
    min = column[0]
    for i in column:
        if i < min:
            min = i
    return min


def find_max(column: pd.Series) -> float:
    """
    Find the maximum value in a pandas Series.
    """
    max = column[0]
    for i in column:
        if i > max:
            max = i
    return max


def calculate_percentile(column: pd.Series, percentile: int) -> float:
    """
    Calculate the given percentile of a pandas Series.
    """
    data_sorted = sorted(column)
    k = (len(data_sorted) - 1) * (percentile / 100)
    f = int(k)
    c = k - f
    if f + 1 < len(data_sorted):
        return round(data_sorted[f] +
                     (data_sorted[f + 1] - data_sorted[f]) * c, 6)
    else:
        return round(data_sorted[f], 6)


def calculate_skewness(column: pd.Series) -> float:
    n = len(column)
    mean = calculate_mean(column)
    var, std = calculate_std(column, mean)
    first_part = n / ((n - 1) * (n - 2))
    second_part = np.sum(((column - mean) / std) ** 3)
    return first_part * second_part


def calculate_kurtosis(column: pd.Series) -> float:
    n = len(column)
    mean = calculate_mean(column)
    var, std = calculate_std(column, mean)
    return (1 / n) * np.sum(((column - mean) / std) ** 4) - 3


def calculate_range(column: pd.Series) -> float:
    """
    Calculate the range of a pandas Series.
    """
    return find_max(column) - find_min(column)
