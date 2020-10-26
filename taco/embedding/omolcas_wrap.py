""" The Base Class for Wrappers. """

import re
import numpy as np

from taco.embedding.postscf_wrap import PostScfWrap
from taco.translate.tools import parse_matrices, triangular2square
from taco.translate.tools import reorder_matrix
from taco.embedding.density import prepare_omolcas_density


class OpenMolcasWrap(PostScfWrap):
    """Base class for Post-SCF from Quantum Chemistry Packages.

    Attributes
    ----------
    scf_wrap : ScfWrap object.
        All information about the scf from which is based.
    energy_dict :  dict
        Container for energy results.
    dms_dict :  dict
        Container for matrices involved in embedding calculation.

    Methods
    -------
    __init__(self, scf_wrap)
    get_density(self)
    format_potential(self, filename)
    write_input(self)
    save_info(self)
    print_embedding_information(self, to_csv)
    export_matrices(self)

    """
    def __init__(self, emb_pot):
        """The wrapper for SCF in QC packages to perform FDET calculations.

        Parameters
        ----------
        emb_pot : dict
            Parameters for the embedding calculation:
            method, basis, x_func, c_func, t_func.

        """
        PostScfWrap.__init__(self, emb_pot)

    def format_potential(self):
        """Reorder embedding potential to be used in OpenMolcas.

        Parameters
        ----------
        filename : str
            Name of the file to be used.

        """
        inprog = self.emb_pot.maininfo["inprog"]
        basis = self.emb_pot.maininfo["basis"]
        atoms = self.emb_pot.maininfo["atoms"]
        vemb_copy = self.emb_pot.compute_embedding_potential()
        ordered_vemb = reorder_matrix(vemb_copy, inprog, 'molcas', basis, atoms)
        np.savetxt('vemb_ordered.txt', ordered_vemb, delimiter='\n')
        # OpenMolcas only reads triangular matrices
        lotri = np.tril(ordered_vemb)
        lt = lotri[np.where(lotri != 0.0)]
        np.savetxt('vemb_ordered_tri.txt', lt, delimiter='\n')

    def write_input(self):
        """Write the input for OpenMolcas."""
        raise NotImplementedError

    @staticmethod
    def prepare_density_file(fname, mo_repr=False):
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
        dm = prepare_omolcas_density(fname, mo_repr)
        return dm

    def get_density(self, fname=None):
        r"""Read Post-SCF density from OpenMolcas.

        Returns
        -------
        dms : np.ndarray(NBas*(NBas+1)/2),
            One-electron density (packed) matrix.
            Off-diagonal elements are double the size:
            :math: `E=\sum_{n<m} D_{mn} H_{mn}`
            Only alpha for restricted case.

        """
        if fname is None:
            raise ValueError("Input filename of DM must be given.")
        # Read from file
        inp = self.prepare_density_file(fname)
        # Re-shape into AO square matrix
        nbas = int(self.emb_pot.maininfo["nbas"])
        dm_inp = triangular2square(inp, nbas)
        # From triangular to square matrix
        dm_out = 0.5*(dm_inp.T + dm_inp)
        np.fill_diagonal(dm_out, np.diag(dm_inp))
        # TODO: Miss reordering
        outprog = self.emb_pot.maininfo["inprog"]
        basis = self.emb_pot.maininfo["basis"]
        atoms = self.emb_pot.maininfo["atoms"]
        dm = reorder_matrix(dm_out, 'molcas', outprog, basis, atoms)
        return dm
