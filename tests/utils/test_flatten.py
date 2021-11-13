import numpy as np
import pytest

from jmspack.utils import flatten


@pytest.fixture
def list_of_lists_test():
    """example set of list of lists"""
    return [
        [x for x in range(10)],
        [x for x in range(10, 20)],
        [x for x in range(20, 30)],
    ]


class TestFlatten:
    """Testing class to test the flatten function."""

    def test_output_matches_list(self, list_of_lists_test):
        """Check if the function output matches what is expected."""
        output_array = np.array(flatten(list_of_lists_test))

        comparison_array = np.array([x for x in range(30)])

        np.testing.assert_array_almost_equal(output_array, comparison_array)
