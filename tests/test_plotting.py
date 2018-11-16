import matplotlib.cm as mcm
import pandas as pd
import pytest

from bevel.plotting import _DivergentBarPlotter

from pandas.testing import assert_frame_equal

@pytest.fixture
def sample_data_even():
    a, b, c = 'a', 'b', 'c'
    return pd.DataFrame.from_dict({
        'group': [a, a, b, b, b, b, c, c, c, c],
        'resps': [1, 2, 3, 4, 1, 2, 3, 2, 2, 3],
    })

@pytest.fixture
def sample_data_odd():
    a, b, c = 'a', 'b', 'c'
    return pd.DataFrame.from_dict({
        'group': [a, a, b, b, b, b, c, c, c, c],
        'resps': [1, 2, 2, 3, 1, 1, 2, 3, 1, 2],
    })

@pytest.fixture
def sample_dbp_odd(sample_data_odd):
    return _DivergentBarPlotter(sample_data_odd, 'group', 'resps')

@pytest.fixture
def sample_dbp_even(sample_data_even):
    return _DivergentBarPlotter(sample_data_even, 'group', 'resps')

class TestDivergentBarPlotter():

    def test_midpoint_default_even(self, sample_dbp_even):
        assert sample_dbp_even.midpoint == 2.5

    def test_midpoint_default_odd(self, sample_dbp_odd):
        assert sample_dbp_odd.midpoint == 2.0
        
    def test_response_label_default(self, sample_dbp_even):
        sample_dbp_even.response_labels == {1: 1, 2: 2, 3: 3, 4: 4}

    def test_compute_bar_sizes_even(self, sample_dbp_even):
        actual = sample_dbp_even._compute_bar_sizes()
        expected = pd.DataFrame([
            {'resps': 1, 'a': -1.00, 'b': -0.50, 'c': -0.50},
            {'resps': 2, 'a': -0.50, 'b': -0.25, 'c': -0.50},
            {'resps': 4, 'a': +0.00, 'b': +0.50, 'c': +0.50},
            {'resps': 3, 'a': +0.00, 'b': +0.25, 'c': +0.50},
        ])
        expected = expected.set_index('resps', drop=True).rename_axis('group', axis='columns')
        assert_frame_equal(actual, expected)
    
    def test_compute_bar_sizes_odd(self, sample_dbp_odd):
        actual = sample_dbp_odd._compute_bar_sizes()
        expected = pd.DataFrame([
            {'resps': 1, 'a': -0.75, 'b': -0.625, 'c': -0.50},
            {'resps': 2, 'a': -0.25, 'b': -0.125, 'c': -0.25},
            {'resps': 3, 'a': +0.25, 'b': +0.375, 'c': +0.50},
            {'resps': 2, 'a': +0.25, 'b': +0.125, 'c': +0.25},
        ])
        expected = expected.set_index('resps', drop=True).rename_axis('group', axis='columns')
        assert_frame_equal(actual, expected)

    def test_compute_bar_sizes_with_fixed_midpoint(self, sample_dbp_even):
        sample_dbp_even.midpoint = 3.1
        actual = sample_dbp_even._compute_bar_sizes()
        expected = pd.DataFrame([
            {'resps': 1, 'a': -1.0, 'b': -0.75, 'c': -1.0},
            {'resps': 2, 'a': -0.5, 'b': -0.50, 'c': -1.0},
            {'resps': 3, 'a': +0.0, 'b': -0.25, 'c': -0.5},
            {'resps': 4, 'a': +0.0, 'b': +0.25, 'c': +0.0},
        ])
        expected = expected.set_index('resps', drop=True).rename_axis('group', axis='columns')
        assert_frame_equal(actual, expected)

    def test_compute_bar_colors(self, sample_dbp_even):
        # mcm.binary is a simple black to white color map  
        # so bar colors sampled from it should be evenly spaced from one to zero
        actual = sample_dbp_even._compute_bar_colors(mcm.binary)
        bar_colors = [pytest.approx(actual[r][0]) for r in sample_dbp_even.response_values]        
        assert bar_colors == [1.0, 2.0 / 3.0, 1.0 /3.0, 0.0]
