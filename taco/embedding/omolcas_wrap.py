""" The Base Class for Wrappers. """

import json
import numpy as np

from taco.embedding.postscf_wrap import PostScfWrap, get_order_lists
from taco.translate.order import transform
from taco.data.cache import data


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
        # Get data from json file in data folder
        jsonfn = data.jfiles['translation']
        with open(jsonfn, 'r') as jf:
            formatdata = json.load(jf)
        inprog = self.emb_pot.maininfo["inprog"]
        basis = self.emb_pot.maininfo["basis"]
        atoms = self.emb_pot.maininfo["atoms"]
        natoms = len(atoms)
        transkey = inprog+'2molcas'
        if not formatdata[transkey]:
            raise KeyError("No translation information available for %s SCF program." % inprog)
        # check that there is info for the basis requested
        if not formatdata[transkey][basis]:
            raise KeyError("The information for %s basis is missing." % basis)
        orders = get_order_lists(atoms, formatdata[transkey][basis])
        vemb_copy = self.emb_pot.compute_embedding_potential()
        ordered_vemb = transform(vemb_copy, natoms, orders)
        np.savetxt('vemb_ordered.txt', ordered_vemb, delimiter='\n')
        # OpenMolcas only reads triangular matrices
        lotri = np.tril(ordered_vemb)
        lt = lotri[np.where(lotri != 0.0)]
        np.savetxt('vemb_ordered_tri.txt', lt, delimiter='\n')

    def write_input(self):
        """Write the input for OpenMolcas."""
        raise NotImplementedError

    def prepare_density_file(self):
        """Prepare density to be read from OpenMolcas."""
        raise NotImplementedError

    def get_density(self, fname=None):
        """Read Post-SCF density from OpenMolcas.

        Returns
        -------
        dms : np.ndarray(NBas*(NBas+1)/2),
            One-electron density (triangular) matrix. Only alpha for restricted
            case.

        """
        if fname is None:
            raise ValueError("Input filename of DM must be given.")
        # Read from file
        inp = np.loadtxt(fname)
        # Re-shape into AO square matrix
        nbas = self.emb_pot.maininfo["nbas"]
        dm_inp = np.zeros((nbas, nbas))
        count = 0
        for i in range(nbas):
            j = i + 1
            dm_inp[i, :j] = inp[count:count+j]
            count += j
        # From triangular to square matrix
        dm_out = dm_inp.T + dm_inp
        np.fill_diagonal(dm_out, np.diag(dm_inp))
        return dm_out
