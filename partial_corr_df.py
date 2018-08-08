import pandas as pd
import numpy as np


def partial_corr(dataframe):
    """
Compute partial correlation matrix

    Computing partial correlations matrix. Takes and returnes pandas.DataFrame objects.
    This uses the correlation matrix inverse approach (cofactors approach) to compute
    the partial correlation.
    The algorithm is detailed here:
        https://en.wikipedia.org/wiki/Partial_correlation#Using_matrix_inversion

    Partial correlation measures the degree of association between two random variables,
    with the effect of a set of controlling random variables removed.
    Like the correlation coefficient, the partial correlation coefficient takes on
    a value in the range from –1 to 1.

        1 conveys a perfect positive linear relationship,
        0 conveys that there is no linear relationship,
        –1 conveys a perfect negative relationship.

    Read more: https://en.wikipedia.org/wiki/Partial_correlation

    Parameters
    ----------
    dataframe : pandas.DataFrame object.
        data for analysis

    Returns
    -------
    partial_corr_df : pandas.DataFrame object
        partial correlation coefficients matrix (partial correlation matrix)
    """
    assert isinstance(dataframe, pd.DataFrame), "{} is not pandas.DataFrame object".format(
        dataframe
    )

    corr_matrix = dataframe.corr()

    class NaNValues(Exception):
        pass

    if any(np.isnan(corr_matrix)):
        raise NaNValues("dataframe.corr() has NaN values")

    names = corr_matrix.index
    names_len = len(names)
    len_range = range(names_len)

    corr_matrix_np = np.array(corr_matrix)
    corr_matrix_inversed = np.linalg.inv(corr_matrix_np)
    partial_corr_np = np.zeros((names_len, names_len))

    for i in len_range:
        for j in len_range:
            if i == j:
                partial_corr_np[i, j] = 1
            else:
                partial_corr_np[i, j] = (
                        -corr_matrix_inversed[i, j] / np.sqrt(corr_matrix_inversed[i, i] * corr_matrix_inversed[j, j])
                )

    partial_corr_df = pd.DataFrame(
        data=partial_corr_np,
        index=names,
        columns=names,
        dtype=None,
        copy=False
    )

    return partial_corr_df
