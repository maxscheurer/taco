"""Tools to handle densities."""

import re
import numpy as np
from taco.translate.tools import parse_matrices


def transform_density_matrix(dm, coeffs):
    """Transform the Density Metrix into a different orbital representation.

    Parameters
    ----------
    dm : np.ndarray
        Density matrix to be transformed.
    coeffs : np.ndarray
        Set of orbital coefficients used for transformation, as follows:

    .. math::
        \gamma_{fin} = \sum_{ij} \chi_i \gamma_{ij} \chi^*_j

    Returns
    -------
    dm_fin : np.ndarray
        Transformed density matrix
    """
    dm_fin = np.einsum('ij,ki,lj->kl', dm, coeffs, coeffs)
    return dm_fin


def prepare_omolcas_density(fname, mo_repr=False):
    """Prepare density to be read from OpenMolcas.

    Parameters
    ----------
    fname : str
        Name of file from where the density is taken.
    mo_repr : bool
        Whether the density matrix to be read is in MO basis.

    Returns
    -------
    dm : np.ndarray
        Density matrix in AO or MO representation.
    """
    if "RUNASCII" in fname:
        if mo_repr:
            hook = {'1dm': re.compile(r'\<(D1mo.)')}
        else:
            hook = {'1dm': re.compile(r'\<(D1ao.)')}
        parsed = parse_matrices(fname, hook, software='molcas')
        return parsed['1dm']
    elif fname.endswith('.h5'):
        pass
    else:
        raise ValueError("""Please specify correctly the name of file,"""
                         """ either containing `RUNASCII` or use a `.h5`""")
