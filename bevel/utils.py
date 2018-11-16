import pandas as pd


def pivot_proportions(df, groups, responses, weights=1):
    """
    Pivot data to show the breakdown of responses for each group.

    Parameters:
      df: a pandas DataFrame with data to be aggregated
      groups: the name of the column containing the groups to partition by
      respones: the name of the column that contains responses to aggregate into proportions
      weights: the statistical weighting associated with each response

    Returns:
      a pandas DataFrame containing the proportion of responses within each group
    """
    
    pivot_data = df[[groups, responses]].assign(weights=weights)
    pivoted_counts = pd.pivot_table(
        pivot_data,
        columns=groups,
        index=responses,
        aggfunc='sum'
    )
    pivoted_counts = pivoted_counts['weights'].sort_index(axis=1)
    return (pivoted_counts / pivoted_counts.sum()).fillna(0)
