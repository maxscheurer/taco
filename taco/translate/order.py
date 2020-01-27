"""Re-order matrices to connect between different programs."""
import numpy as np


def get_sort_list(natoms, orders):
    """Get the molecule sord list for transforming matrices

    Parameters:
    -----------
    natoms : int
        Number of atoms in the molecule.
    orders : list[list[ N atoms (int)]]
        Order of basis functions compared to reference program,
        one list per atom.

    Returns:
    --------
    List with the whole molecule order indices.

    """
    sort = []
    offset = 0
    for iatom in range(natoms):
        for n in orders[iatom]:
            sort.append(n + offset)
        offset += len(orders[iatom])
    return sort


def transform(inparr, natoms, orders):
    """Perform symmetric transformation of a matrix.
    The symmetric transformation of a matrix :math:`\\mathbf{X}` uses a
    rearranged identity matrix :math:`\\mathbf{P}` such that the working
    equation is:
    :math:`\\mathbf{P}^{T}\\cdot \\mathbf{X} \\cdot\\mathbf{P}`

    Parameters:
    -----------
    inparr : np.ndarray
        Input matrix :math:`\\mathbf{X}`.

    Returns:
    --------
    Q : np.ndarray
        Transformed matrix according to target format.

    """
    # -------------------
    # Q = P.T * X * P
    # -------------------
    sort_list = get_sort_list(natoms, orders)
    nAO = inparr.shape[0]
    idarr = np.identity(nAO)
    # Do transformation
    P = idarr[:, sort_list]  # black magic: rearrange columns of ID matrix
    M = np.dot(P.T, inparr)
    Q = np.dot(M, P)
    return Q
