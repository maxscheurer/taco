"""Some useful tools to comunicate with other QC software."""

import re
import json
import numpy as np

from taco.data.cache import data
from taco.translate.order import transform


def parse_matrix_molcas(lines, nline, file_type='runascii'):
    """Parse a matrix from RUNASCII OpenMolcas file.

    Parameters
    ----------
    lines :  list
        From readlines object.
    nline : int
        Line number of identifier.
    file_type : str
        Type of file from where it is parsed, determines how
        to define the matrix.

    Returns
    -------
    matrix : np.ndarray
        Matrix in one dimension: (MxN,) shape.
    """
    if not isinstance(lines, list):
        raise TypeError("lines should be given as a list.")
    if not isinstance(nline, int):
        raise TypeError("nline must be int.")
    if file_type == 'runascii':
        iline = nline + 1
        dimension = int(lines[iline].strip().split()[0])
        iline += 1
    elif file_type == 'orbitals':
        dimension = int(lines[4].strip())
        iline = nline + 1
    elif file_type == 'h5':
        raise NotImplementedError
    else:
        raise ValueError('`file_type` is not correct or not available.')

    # Now read numbers
    matrix = []
    while len(matrix) < dimension:
        data = lines[iline].strip().split()
        for d in data:
            if 'D' in d:
                d = d.replace("D", "E")
            matrix.append(float(d))
        iline += 1
    return np.array(matrix)


def parse_matrices(data, hooks, software, file_type=None):
    """Parse many matrices from file.

    Parameters
    ----------
    data : str or list
        If str, the filename to be parsed. If list, the list
        of lines of strings is expected.
    hooks : dict
        Dictionary with the name of the element to parse
        and the hooks or identifier.
    software : str
        Name of software from which file is parsed.

    Returns
    -------
    parsed : dict
        Parsed objects.
    """
    if not isinstance(hooks, dict):
        raise TypeError("`hooks` should be given as a dictionary.")
    parsed = {}
    if isinstance(data, list):
        lines = data
    elif isinstance(data, str):
        with open(data, 'r') as thefile:
            lines = thefile.readlines()
    else:
        raise TypeError('`data` must be either str or list of str.')
    for n, line in enumerate(lines):
        for key, hook in hooks.items():
            match = hook.search(line)
            if match:
                if software == 'molcas':
                    if file_type:
                        parsed[key] = parse_matrix_molcas(lines, n, file_type)
                    else:
                        parsed[key] = parse_matrix_molcas(lines, n)
                else:
                    raise NotImplementedError
    return parsed


def parse_orbitals_molcas(fname):
    """ Parse orbitals from OMolcas orbital files.

    Parameters
    ----------
    fname : str
        Filename of orbitals file.

    Returns
    -------
    orbs : np.array
        N x N matrix with orbital coefficients orbs[i, :] is each orbital.
    """
    with open(fname, 'r') as thefile:
        lines = thefile.readlines()
    norbs = int(lines[4].strip())  # Assuming it's always same format
    orbs = np.zeros((norbs, norbs))
    hooks = {}
    for i in range(1, norbs+1):
        string = r'\* ORBITAL    1' + '{:>5}'.format(i)
        hooks['orbital%d' % i] = re.compile(string)
    parsed = parse_matrices(lines, hooks, 'molcas', file_type='orbitals')
    for i in range(norbs):
        orbs[i, :] = parsed['orbital%d' % (i+1)]
    return orbs


def triangular2square(trimat, n):
    """Make a square symmetric matrix from a triangular one.

    Parameters
    ----------
    trimat : np.ndarray((n+1)*n/2),)
        Triangular matrix in one dimension.
    n : int
        Dimension of the square matrix: n x n.

    Returns
    -------
    sqmat : np.ndarray((n,n))
        Full square symmetric matrix
    """
    if not isinstance(trimat, np.ndarray):
        raise TypeError("`trimat` must be a np.ndarray.")
    if not isinstance(n, int):
        raise TypeError('`n` must be int.')
    trilen = n*(n+1)/2
    if trimat.size != trilen:
        raise ValueError("The shape of the triangular matrix does not match with final length n.")
    sqmat = np.zeros((n, n))
    count = 0
    for i in range(n):
        j = i + 1
        sqmat[i, :j] = trimat[count:count+j]
        count += j
    return sqmat


def get_order_lists(atoms, basis_dict):
    """Get list of orders for matrix re-ordering.

    Parameters
    ----------
    atoms : np.darray(int)
        Atoms in molecule/fragment.
    basis_dict : dict
        Known orders for each row/group in the periodic table.

    Returns
    -------
    orders : list[list[],]
        List with order for each atom.
    """
    if not isinstance(atoms, np.ndarray):
        raise TypeError("`atoms` must be provided in a np.darray.")
    if atoms.dtype != int:
        raise NotImplementedError('For now, atomic numbers are accepted only.')
    orders = []
    for atom in atoms:
        if atom < 3:
            orders.append(basis_dict['first'])
        elif 2 < atom < 11:
            orders.append(basis_dict['second'])
        elif 11 < atom < 19:
            orders.append(basis_dict['third'])
        else:
            raise NotImplementedError('At the moment only first and second row elements are available.')
    return orders


def reorder_matrix(inmat, inprog, outprog, basis, atoms):
    """Re-order matrix to fit some other program format.

    Parameters
    ---------
    inmat : np.ndarray((n,n))
        Square symmetric matrix to be re-ordered.
    inprog, outprog :  str
        Name of the programs to connect, all lowercase.
    basis : str
        Basis set name.
    atoms : np.ndarray
        Atomic numbers of molecule/fragment.

    Returns
    -------
    ordered : np.ndarray((n,n))
        Re-ordered square symmetric matrix.
    """
    if not isinstance(inmat, np.ndarray):
        raise TypeError("`inmat` must be a np.ndarray object.")
    if not isinstance(inprog, str):
        raise TypeError("`inprog` must be a string.")
    if not isinstance(outprog, str):
        raise TypeError("`outprog` must be a string.")
    if not isinstance(basis, str):
        raise TypeError("`basis` must be a string.")
    if atoms.dtype != int:
        raise TypeError("`atoms` must be an array with integer numbers.")
    # Get data from json file in data folder
    jsonfn = data.jfiles['translation']
    with open(jsonfn, 'r') as jf:
        formatdata = json.load(jf)
    natoms = len(atoms)
    transkey = inprog+'2'+outprog
    if not formatdata[transkey]:
        raise KeyError("No translation information available for %s SCF program." % inprog)
    # check that there is info for the basis requested
    if not formatdata[transkey][basis]:
        raise KeyError("The information for %s basis is missing." % basis)
    orders = get_order_lists(atoms, formatdata[transkey][basis])
    ordered = transform(inmat, natoms, orders)
    return ordered


if __name__ == '__main__':
    # Molcas example
    fname = 'RUNASCII'
    hooks = {'1dm': re.compile(r'\<(D1ao.)'),
             'dipole': re.compile(r'\<(Dipole moment.)'),
             'fock': re.compile(r'\<(FockOcc.)'),
             'scf_orbs': re.compile(r'\<(SCF orbitals.)'),
             }
    p = parse_matrices(fname, hooks, 'molcas')
