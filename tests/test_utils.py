import pytest

from bevel.utils import *

from pandas.testing import assert_frame_equal


@pytest.fixture
def sample_response_data():
    a, b, c = 'a', 'b', 'c'
    x, y, z = 'x', 'y', 'z'
    return pd.DataFrame.from_dict({
        'groups_a': [a, a, b, b, b, b, b, c, c, c, c, c],
        'groups_x': [x, x, x, x, x, y, y, y, y, y, z, z],
        'response': [1, 2, 1, 2, 3, 4, 4, 2, 3, 3, 4, 4],
        'weights_': [1, 4, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
    })


def test_pivot_proportions(sample_response_data):
    actual = pivot_proportions(sample_response_data, 'groups_a', 'response')
    expected = pd.DataFrame([
        {'response': 1, 'a': 0.5, 'b': 0.2, 'c': 0.0},
        {'response': 2, 'a': 0.5, 'b': 0.2, 'c': 0.2},
        {'response': 3, 'a': 0.0, 'b': 0.2, 'c': 0.4},
        {'response': 4, 'a': 0.0, 'b': 0.4, 'c': 0.4},
    ])
    expected = expected.set_index('response', drop=True).rename_axis('groups_a', axis='columns')
    assert_frame_equal(actual, expected)


def test_pivot_proportions_with_weights(sample_response_data):
    actual = pivot_proportions(
        sample_response_data, 
        'groups_a', 
        'response', 
        weights=sample_response_data['weights_']
    )
    expected = pd.DataFrame([
        {'response': 1, 'a': 0.2, 'b': 0.2, 'c': 0.0},
        {'response': 2, 'a': 0.8, 'b': 0.2, 'c': 0.2},
        {'response': 3, 'a': 0.0, 'b': 0.2, 'c': 0.4},
        {'response': 4, 'a': 0.0, 'b': 0.4, 'c': 0.4},
    ])
    expected = expected.set_index('response', drop=True).rename_axis('groups_a', axis='columns')
    assert_frame_equal(actual, expected)
