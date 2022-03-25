import numpy as np
from numpy import isnan, nan, logical_not, logical_or
import sys


def exclude_nan(x, y):
    """
    Exclude NaN values if either entry in a pair of vectors has NaN
    """
    idx = logical_not(logical_or(isnan(x), isnan(y)))
    x = x[idx]
    y = y[idx]
    n = len(x)
    return [x, y, n]


def compute_icc(x, y):
    """
    This function computes the intra-class correlation (ICC) of the
    two classes represented by the x and y numpy vectors.
    """
    if all(x == y):
        return 1

    [x, y, n] = exclude_nan(x, y)

    ## Need at least 3 data points to compute this
    if n < 3:
        return nan

    Sx = sum(x)
    Sy = sum(y)
    Sxx = sum(x*x)
    Sxy = sum((x+y)**2) / 2
    Syy = sum(y*y)

    fact = ((Sx + Sy)**2) / (n*2)
    SS_tot = Sxx + Syy - fact
    SS_among = Sxy - fact
    SS_error = SS_tot - SS_among

    MS_error = SS_error / n
    MS_among = SS_among / (n-1)

    ICC = (MS_among - MS_error) / (MS_among + MS_error)

    return ICC
