""" The Base Class for Wrappers. """

import json
import numpy as np

from taco.embedding.postscf_wrap import PostScfWrap


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
    translate_potential(self, filename)
    write_input(self)
    save_info(self)
    print_embedding_information(self, to_csv)
    export_matrices(self)

    """
    def __init__(self, scf_wrap):
        """The wrapper for SCF in QC packages to perform FDET calculations.

        Parameters
        ----------
        frag_args : dict
            Parameters for individual fragments:
            molecule, method, basis.
        emb_args : dict
            Parameters for the embedding calculation:
            method, basis, x_func, c_func, t_func.

        """
        PostScfWrap.__init__(self, scf_wrap)

    def translate_potential(self, filename=None):
        """Reorder embedding potential to be used in OpenMolcas.

        Parameters
        ----------
        filename : str
            Name of the file to be used.

        """
        # Get data from json file in data folder
        jsonfn = os.path.join(datafolder, 'translation.json')
        with open(jsonfn, 'r') as jf:
            data = json.load(jf)
        inprog = get_scf_program(self.scf_wrap)
        transkey = inprog+'2molcas'
        if not data[transkey]:
            raise KeyError("No translation information available for %s SCF program." % inprog)
        # check that there is info for the basis requested
        basis = self.scf_wrap.mol0.basis
        atoms = self.scf_wrap.mol0.atom
        natoms = len(atoms)
        if not data[transkey][basis]:
            raise KeyError("The information for %s basis is missing." % basis)
        orders = get_order_lists(atoms, data[transkey][basis])
        vemb_copy = np.copy(self.scf_wrap.vemb_dict["vemb"])
        ordered_vemb = transform(vemb_copy, natoms, orders)
        np.savetxt('vemb_ordered.txt', ordered_vemb, delimiter='\n')
        # OpenMolcas only reads triangular matrices
        lotri = np.tril(ordered_vemb)
        lt = lotri[np.where(abs(lotri - 0.0) > 1e-9)]
        np.savetxt('vemb_ordered_tri.txt', lt, delimiter='\n')

    def write_input(self):
        """Write the input for OpenMolcas."""
        return

    def get_density(self):
        """Read Post-SCF density from OpenMolcas.

        Returns
        -------
        dms : list[np.ndarray,]
            One-electron density matrices. Only alpha for restricted
            case.

        """
        raise NotImplementedError
