import pandas as pd


def pivot_proportions(df, groups, responses, weights=1):
    pivot_data = df[[groups, responses]].assign(weights=weights)
    pivoted_counts = pd.pivot_table(
        pivot_data,
        columns=groups,
        index=responses,
        aggfunc='sum'
    )
    pivoted_counts = pivoted_counts['weights'].sort_index(axis=1)
    return (pivoted_counts / pivoted_counts.sum()).fillna(0)
