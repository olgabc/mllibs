"""
Computing partial correlations matrix. Takes and returnes pandas.DataFrame objects.
This uses the correlation matrix inverse approach (cofactors approach) to compute the partial correlation The
algorithm is detailed here:

    https://en.wikipedia.org/wiki/Partial_correlation#Using_matrix_inversion

"""

import pandas as pd
import numpy as np


def partial_corr_df(dataframe):
    assert isinstance(dataframe, pd.DataFrame), "{} is not pandas.DataFrame object".format(
        dataframe
    )

    corr_matrix = dataframe.corr()

    names = corr_matrix.index
    names_len = len(names)
    len_range = range(names_len)

    corr_matrix_np = np.array(corr_matrix)
    cofactors_matrix = np.linalg.inv(corr_matrix_np)
    partial_corr_np = np.zeros((names_len, names_len))

    for i in len_range:
        for j in len_range:
            if i == j:
                partial_corr_np[i, j] = 1
            else:
                partial_corr_np[i, j] = (
                        -cofactors_matrix[i, j] / np.sqrt(cofactors_matrix[i, i] * cofactors_matrix[j, j])
                )

    partial_corr_df = pd.DataFrame(
        data=partial_corr_np,
        index=names,
        columns=names,
        dtype=None,
        copy=False
    )

    return partial_corr_df
