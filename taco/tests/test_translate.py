"""
Test translate functions.
"""
import re
import pytest
import numpy as np

from taco.translate.order import get_sort_list
from taco.translate.order import transform
from taco.translate.tools import triangular2square, reorder_matrix
from taco.translate.tools import parse_matrix_molcas, parse_matrices
from taco.testdata.cache import cache


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


def test_parse_matrix_molcas():
    """Reading matrices from molcas RUNASCII."""
    n0 = 'n'
    lines0 = 'line'
    with pytest.raises(TypeError):
        parse_matrix_molcas(lines0, n0)
    with pytest.raises(TypeError):
        parse_matrix_molcas([lines0], n0)
    lines = [' oasjnvriaksnc', '  3       2',
             '14.980    1230.0      984.0128']
    n = 0
    matrix = parse_matrix_molcas(lines, n)
    np.testing.assert_allclose(matrix, np.array([14.980, 1230.0, 984.0128]))


def test_parse_matrices_molcas():
    """Read matrices parser, OMolcas example."""
    fname = cache.files["molcas_runascii_co_h2o_cc-pvdz"]
    hook0 = 'hook'
    with pytest.raises(TypeError):
        parse_matrices(fname, hook0, 'molcas')
    hook = {'dipole': re.compile(r'\<(Dipole moment.)')}
    with pytest.raises(NotImplementedError):
        parse_matrices(fname, hook, 'qchem')
    parsed = parse_matrices(fname, hook, 'molcas')
    ref = np.array([0.165830742659920816, -0.228990590320634624e-01,
                    -0.196753869675605486e-01])
    np.testing.assert_allclose(ref, parsed['dipole'])


def test_triangular2square():
    """Test function to make full square matrix."""
    mat0 = [0, 1, 2, 3]
    n0 = 'number'
    mat1 = np.arange(0, 9)
    n1 = 4 
    with pytest.raises(TypeError):
        triangular2square(mat0, n1)
    with pytest.raises(TypeError):
        triangular2square(mat1, n0)
    with pytest.raises(ValueError):
        triangular2square(mat1, n1)
    mat = np.arange(0, 10)
    square = triangular2square(mat, n1)
    ref = np.array([[0, 0, 0, 0],
                    [1, 2, 0, 0],
                    [3, 4, 5, 0],
                    [6, 7, 8, 9]])
    np.testing.assert_allclose(square, ref)



def test_reoder_matrix():
    """Test re-ordering function for matrices."""
    inmat0 = [0, 1, 2]
    inprog0 = 3
    basis0 = 0
    atoms0 = np.array([0.2, 1.5])
    inmat1 = np.array([0, 1, 2])
    inprog1 = 'pyscf'
    basis1 = 'cc-pvdz'
    basis2 = 'cc-pvtz'
    atoms1 = np.array([2])
    with pytest.raises(TypeError):
        reorder_matrix(inmat0, inprog1, inprog1, basis1, atoms1)
    with pytest.raises(TypeError):
        reorder_matrix(inmat1, inprog0, inprog1, basis1, atoms1)
    with pytest.raises(TypeError):
        reorder_matrix(inmat1, inprog1, inprog0, basis1, atoms1)
    with pytest.raises(TypeError):
        reorder_matrix(inmat1, inprog1, inprog1, basis0, atoms1)
    with pytest.raises(TypeError):
        reorder_matrix(inmat1, inprog1, inprog1, basis1, atoms0)
    outprog0 = 'gaussian'
    outprog1 = 'molcas'
    with pytest.raises(KeyError):
        reorder_matrix(inmat1, inprog1, outprog0, basis1, atoms1)
    with pytest.raises(KeyError):
        reorder_matrix(inmat1, inprog1, outprog1, basis2, atoms1)
    ref = np.array([[0, 1, 3, 6, 10],
                    [1, 2, 4, 7, 11],
                    [3, 4, 5, 8, 12],
                    [6, 7, 8, 9, 13],
                    [10, 11, 12, 13, 14]])
    finmat = reorder_matrix(ref, inprog1, outprog1, basis1, atoms1)
    np.testing.assert_allclose(finmat, ref)
    

if __name__ == "__main__":
    test_get_sort_list()
    test_transform()
    test_parse_matrix_molcas()
    test_parse_matrices_molcas()
    test_triangular2square()
    test_reoder_matrix()
