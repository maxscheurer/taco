""" The Base Class for Wrappers. """


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
        self.energy_dict = {}
        self.vemb_dict = {}
        self.emb_args = emb_args
        self.mol0 = None
        self.mol1 = None
        self.mmethod0 = None
        self.mmethod1 = None

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

    def print_embedding_information(self, to_csv=False):
        """Print all the results from the calculation.

        Parameters
        ----------
        to_csv : bool
            Whether to save the information on a file or not.
            Default prints only on the screen.
        """
        raise NotImplementedError

    def export_matrices(self):
        """Save all matrices into files."""
        raise NotImplementedError
