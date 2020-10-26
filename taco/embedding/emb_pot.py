"""PySCF Utilities for Embedding calculations."""

import numpy as np
from pyscf.dft.libxc import xc_type


class EmbPotBase():
    """Base class for embedding potentials.

    Attributes
    ----------
    mol0, mol1 :
        Molecule objects.
    main_info : dict
        Main information to be used in other places:
        frag0_atoms, frag0_basis
    vemb_dict :  dict
        Container for matrices and energies involved in embedding calculation.

    Methods
    -------
    __init__(self, frag0_args, frag1_args, emb_args)
    check_emb_arguments(emb_args)
    save_maininfo(self)
    assign_dm(self, nfrag, dm)
    compute_coulomb_potential(self)
    compute_attraction_potential(self)
    compute_nad_potential(self)
    compute_embedding_potential(self)

    """
    def __init__(self, mol0, mol1, emb_args):
        """Embedding potential Object.

        Parameters
        ----------
        mol0, mol1 : Depending on the program
            Molecule objects.
        dm0, dm1 : np.ndarray(NAO,NAO)
            One-electron density matrices.
        emb_args : dict
            Parameters for the embedding calculation:
            x_func, c_func, t_func.

        """
        self.check_emb_arguments(emb_args)
        self.emb_args = emb_args
        self.mol0 = mol0
        self.mol1 = mol1
        self.dm0 = None
        self.dm1 = None
        self.vemb_dict = {}

    @staticmethod
    def check_emb_arguments(emb_args):
        if not isinstance(emb_args, dict):
            raise TypeError("emb_args must be a dictionary with embedding arguments.")
        if emb_args['xc_code'] is None:
            raise KeyError("Missing to specify `xc_code` in emb_args.")
        else:
            xctype = xc_type(emb_args['xc_code'])
            if xctype not in ['LDA', 'GGA']:
                raise NotImplementedError("Only `LDA` and `GGA` functionals are available")
        if 't_code' not in emb_args:
            raise KeyError("Missing to specify `t_code` in emb_args.")
        else:
            xctype = xc_type(emb_args['t_code'])
            if xctype not in ['LDA', 'GGA']:
                raise NotImplementedError("Only `LDA` and `GGA` kinetic energy funcitonals available")

    def assign_dm(self, nfrag, dm):
        """Assign matrix to object attribite.

        Parameters
        ----------
        nfrag :  int
            to which fragment/molecule the DM corresponds.

        """
        if nfrag not in [0, 1]:
            raise ValueError("Only 0 or 1 are valid values for nfrag.")
        if not isinstance(dm, np.ndarray):
            raise TypeError("Density matrix must be a np.ndarray.")
        if nfrag == 0:
            self.dm0 = dm
        else:
            self.dm1 = dm

    def save_maininfo(self, mol0):
        """Save in a dictionary basic information of mol0 in a simple format."""
        raise NotImplementedError

    def compute_coulomb_potential(self):
        """Compute the electron-electron repulsion potential.

        Returns
        -------
        v_coul : np.ndarray(NAO,NAO)
            Coulomb repulsion potential.

        """
        raise NotImplementedError

    def compute_attraction_potential(self):
        """Compute the nuclei-electron attraction potentials.

        Returns
        -------
        v0_nuc1 : np.ndarray(NAO,NAO)
            Attraction potential between electron density of fragment0
            and nuclei of fragment1.
        v1_nuc0 : np.ndarray(NAO,NAO)
            Attraction potential between electron density of fragment0
            and nuclei of fragment1.

        """
        raise NotImplementedError

    def compute_nad_potential(self):
        """Compute the non-additive potentials and energies.

        Returns
        -------
        vxc_nad : np.ndarray(NAO,NAO)
            Non-additive Exchange-Correlation + Kinetic potential.

        """
        raise NotImplementedError

    def compute_embedding_potential(self, dm0, dm1):
        """Compute embedding potential.

        Parameters
        ----------
        dm0, dm1 : np.ndarray(NAO,NAO)
            One-electron density matrices.

        Returns
        -------
        vemb : np.ndarray
            Embedding potential as a Fock-like matrix.

        """
        raise NotImplementedError
