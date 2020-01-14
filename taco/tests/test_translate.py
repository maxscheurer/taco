"""
Test translate functions.
"""
import numpy as np

from taco.translate.order import get_sort_list
from taco.translate.order import transform


def test_get_sort_list():
    """Test sorting list function."""
    natoms = 3
    orders = [[0, 1, 2, 3], [0, 1, 2, 3, 6, 4, 7, 5, 8, 9],
              [0, 1, 2, 3, 6, 4, 7, 5, 8, 9]]
    ref_list = [0, 1, 2, 3, 4, 5, 6, 7, 10, 8, 11, 9, 12, 13,
                14, 15, 16, 17, 20, 18, 21, 19, 22, 23]
    sort_list = get_sort_list(natoms, orders)
    assert ref_list == sort_list


def test_transform():
    """Test function to reorder matrices."""
    ref = np.arange(1, 26, 1).reshape(5, 5)
    ref = (ref + ref.T) - np.diag(ref.diagonal())
    # Original array
    # [[ 1  8 14 20 26]
    # [ 8  7 20 26 32]
    # [14 20 13 32 38]
    # [20 26 32 19 44]
    # [26 32 38 44 25]]
    ref_ordered = np.array([[1,  8,  14, 26, 20],
                            [8,  7,  20, 32, 26],
                            [14, 20, 13, 38, 32],
                            [26, 32, 38, 25, 44],
                            [20, 26, 32, 44, 19]])
    atoms = 2
    orders = [[0, 1], [0, 2, 1]]
    result = transform(ref, atoms, orders)
    assert np.allclose(ref_ordered, result)


if __name__ == "__main__":
    test_get_sort_list()
    test_transform()
