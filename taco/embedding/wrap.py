""" The Base Class for Wrappers. """

import numpy as np


class QCWrap():
    """Base class for Quantum Chemistry Packages.

    Attributes
    ----------
    energy_dict :  dict
        Container for energy results.
    vemb_dict :  dict
        Container for matrices involved in embedding calculation.
    mol0, mol1 :
        Molecule objects.
    method0, method1 :
        Method objects.

    Methods
    -------
    __init__(self, frag0_args, frag1_args, emb_args)
    create_fragments(self, frag0_args, frag1_args)
    compute_embedding_potential(self)
    run_embedding(self)
    save_info(self)
    print_embedding_information(self, to_csv)
    export_matrices(self)

    """
    def __init__(self, frag0_args, frag1_args, emb_args):
        """The wrapper for QC packages to perform FDET calculations.

        Parameters
        ----------
        frag_args : dict
            Parameters for individual fragments:
            molecule, method, basis.
        emb_args : dict
            Parameters for the embedding calculation:
            method, basis, x_func, c_func, t_func.

        """
        if not isinstance(frag0_args, dict):
            raise TypeError("Fragment 0 arguments must be provided as a dictionary.")
        if not isinstance(frag1_args, dict):
            raise TypeError("Fragment 1 arguments must be provided as a dictionary.")
        if not isinstance(emb_args, dict):
            raise TypeError("Arguments for the embedding calculation must be provided as a dictionary.")
        self.energy_dict = {}
        self.vemb_dict = {}
        self.emb_args = emb_args
        self.mol0 = None
        self.mol1 = None
        self.mmethod0 = None
        self.mmethod1 = None

    def check_emb_arguments(self, args):
        self.check_basic_arguments(args)
        if not 'xc_code' in args:
            raise KeyError("Missing to specify `xc_code` in emb_args.")
        if not 't_code' in args:
            raise KeyError("Missing to specify `t_code` in emb_args.")

    @staticmethod
    def check_basic_arguments(args):
        if not 'mol' in args:
            raise KeyError("Missing to specify `molecule`.")
        if not 'method' in args:
            raise KeyError("Missing to specify `method`.")
        if not 'basis' in args:
            raise KeyError("Missing to specify `basis`.")

    def create_fragments(self, frag0_args, frag1_args):
        """Save fragment information.

        Parameters
        ----------
        frag_args : dict
            Parameters for individual fragments:
            molecule, method, basis.
        """
        raise NotImplementedError

    def compute_embedding_potential(self):
        """Compute embedding potential.

        Returns
        -------
        vemb : np.ndarray
            Embedding potential as a Fock-like matrix.

        """
        raise NotImplementedError

    def run_embedding(self):
        """Run FDET embedding calculation."""
        raise NotImplementedError

    def save_info(self):
        """Save information after embedding calculation."""
        raise NotImplementedError

    def print_embedding_information(self, to_csv=False):
        """Print all the results from the calculation.

        Parameters
        ----------
        to_csv : bool
            Whether to save the information on a file or not.
            Default prints only on the screen.
        """
        # Print Energy Table
        line = '='*50
        print(line)
        print("{:<50}".format("FDET Results"))
        print(line)
        for label in self.energy_dict:
            num = self.energy_dict[label]
            print("{:<30} {:>16.12f}".format(label, num))
        if to_csv:
            from pandas import DataFrame
            columns = list(self.energy_dict)
            df = DataFrame(self.energy_dict, columns=columns, index=[0])
            df.to_csv("embedding_energies.csv", header=True, index=None)

    def export_matrices(self):
        """Save all matrices into files."""
        for element in self.vemb_dict:
            np.savetxt(element+".txt", self.vemb_dict[element])
