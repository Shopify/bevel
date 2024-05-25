import matplotlib.cm as mcm
import pandas as pd
import numpy as np

from bevel.utils import pivot_proportions


class _DivergentBarPlotter():

    def __init__(self, df, group, response, weights=1, midpoint=None, response_labels=None, cmap=mcm.summer):
        self.proportions = pivot_proportions(df, group, response, weights)
        self.response_values = self.proportions.index.values
        self.min_response = min(self.response_values)
        self.max_response = max(self.response_values)
        self.group_values = self.proportions.columns
        self.midpoint = midpoint
        self.response_labels = response_labels
        self.bar_colors = self._compute_bar_colors(cmap)


    @property
    def midpoint(self):
        return self._midpoint


    @midpoint.setter
    def midpoint(self, value):
        self._midpoint = value or (self.min_response + self.max_response) / 2.0


    @property
    def response_labels(self):
        return self._response_labels


    @response_labels.setter
    def response_labels(self, value):
        self._response_labels = value or dict(zip(self.response_values, self.response_values))


    def _compute_bar_sizes(self):

        def accumulate(df, ascending=True):
            return df._append(middle).sort_index(ascending=ascending).cumsum().iloc[::-1]
        
        middle = self.proportions[self.proportions.index == self.midpoint] * 0.5    
        right_bars = accumulate(self.proportions[self.proportions.index > self.midpoint])
        left_bars = -1.0 * accumulate(self.proportions[self.proportions.index < self.midpoint], False)
        return left_bars._append(right_bars)


    def _compute_bar_colors(self, cmap):
        response_range = self.response_values - self.min_response
        color_samples = [cmap(x / max(response_range)) for x in response_range]
        return dict(zip(self.response_values, color_samples))


    def _label_bars(self, ax):
        colors_to_label = [self.bar_colors[self.min_response], self.bar_colors[self.max_response]]
        label_patches = [patch for patch in ax.patches if patch.get_fc() in colors_to_label]
        
        for patch in label_patches:
            height = patch.get_height()
            width = patch.get_width()
            text = '{:3.1f}%'.format(abs(width*100)) if width != 0 else '' 
            ax.text(patch.get_x() + width, patch.get_y() + height / 2.0, text, ha='center', va='center')


    def _format_axes(self, ax):
        ax.set_yticklabels(self.group_values)
        ax.axvline(0)

        vals = ax.get_xticks()
        ax.set_xticklabels(['{:3.0f}%'.format(abs(x*100)) for x in vals])


    def _add_legend(self, ax):
        import matplotlib.patches as mp
        legend_handles = [
            mp.Patch(label=self.response_labels[l], color=self.bar_colors[l]) for l in self.response_values
        ]
        ax.legend(handles=legend_handles, loc='best')


    def plot(self):
        import matplotlib.pyplot as plt

        divergent_bars = self._compute_bar_sizes()
        
        ax = plt.gca()
        for index, row in divergent_bars.iterrows():
            ax.barh(self.group_values, row, color=self.bar_colors[index])

        self._label_bars(ax)
        self._format_axes(ax)
        self._add_legend(ax)
        return ax
        

def divergent_stacked_bar(df, group, response, weights=1, midpoint=None, response_labels=None, cmap=mcm.summer):
    """
    Aggregate data from a dataframe and plot the results as a divergent stacked bar chart

    Parameters:
      df: a pandas DataFrame with data to be aggregated
      groups: the name of the column containing the groups to partition by
      responses: the name of the column that contains responses for aggregation
      weights: the statistical weighting associated with each response
      midpoint: the middle value of the ordinal responses where the graph is centered
      response_labels: a dictionary between the ordinal resposnes and their labels
      cmap: a matplotlib colormap object from which bar colours are sampled

    Returns:
        ax: a pyplot axis object containing the graph
    """
    return _DivergentBarPlotter(df, group, response, weights, midpoint, response_labels, cmap).plot()
