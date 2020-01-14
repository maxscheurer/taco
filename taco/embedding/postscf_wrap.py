""" The Base Class for Wrappers. """

import numpy as np

from taco.embedding.scf_wrap import ScfWrap


class PostScfWrap():
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
    check_scf_wrap(self)
    get_density(self)
    translate_potential(self, filename)
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
        if not isinstance(scf_wrap, ScfWrap):
            raise TypeError("scf_wrap must be an instance of the ScfWrap class.")
        self.check_scf_wrap(scf_wrap)
        self.energy_dict = {}
        self.dms_dict = {}

    def check_scf_wrap(self, scf_wrap):
        """Confirm that the scf_wrap has all we need or compute it."""
        if not scf_wrap.vemb_dict:
            scf_wrap.run_embedding()
        if not scf_wrap.energy_dict:
            raise ValueError("Something went wrong in the embedding calculation.")
        nad_function = getattr(scf_wrap, "compute_nad_terms", None)
        if not callable(nad_function):
            raise AttributeError("The ScfWrap class should have a compute_nad_terms method.")
        self.scf_wrap = scf_wrap

    def translate_potential(self, filename=None):
        """Reorder embedding potential to be used in QC program.

        Parameters
        ----------
        filename : str
            Name of the file to be used.

        """
        raise NotImplementedError

    def get_density(self):
        """Read Post-SCF density from QC program.

        Returns
        -------
        dms : list[np.ndarray,]
            One-electron density matrices. Only alpha for restricted
            case.

        """
        raise NotImplementedError

    def save_info(self):
        """Save information after embedding calculation."""
        # Get densities from methods
        dm0_final = self.dms_dict["dm0_final"]
        dm1 = self.scf_wrap.vemb_dict["dm1_ref"]
        # Get non-additive information
        # Final density functionals
        final_vnad = self.scf_wrap.compute_nad_terms(self.scf_wrap.emb_method.mol_pyscf,
                                                     self.scf_wrap.mol1, dm0_final,
                                                     dm1, self.scf_wrap.emb_args)
        self.energy_dict["exc_nad_final"] = final_vnad[0]
        self.energy_dict["et_nad_final"] = final_vnad[1]
        v_nad_xc_final = final_vnad[2]
        v_nad_t_final = final_vnad[3]
        int_ref_xc = self.scf_wrap.energy_dict["int_emb_xc"]
        int_ref_t = self.scf_wrap.energy_dict["int_emb_t"]
        self.energy_dict["int_final_xc"] = np.einsum('ab,ba', v_nad_xc_final, dm0_final)
        self.energy_dict["int_final_t"] = np.einsum('ab,ba', v_nad_t_final, dm0_final)
        # Linearization terms
        int_emb_xc = np.einsum('ab,ba', self.scf_wrap.vemb_dict["v_nad_xc_final"], dm0_final)
        int_emb_t = np.einsum('ab,ba', self.scf_wrap.vemb_dict["v_nad_t_final"], dm0_final)
        self.energy_dict["int_emb_xc"] = int_emb_xc
        self.energy_dict["int_emb_t"] = int_emb_t
        self.energy_dict["deltalin"] = (int_emb_xc - int_ref_xc) + (int_emb_t - int_ref_t)

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
        line2 = '-'*50
        print(line)
        print("{:<50}".format("FDET Results"))
        print(line)
        print("{:<50}".format("FDET SCF"))
        print(line2)
        self.scf_wrap.print_embedding_information(to_csv)
        print(line2)
        print("{:<50}".format("FDET Post-SCF"))
        print(line2)
        for label in self.energy_dict:
            num = self.energy_dict[label]
            print("{:<30} {:>16.12f}".format(label, num))
        if to_csv:
            from pandas import DataFrame
            columns = list(self.energy_dict)
            df = DataFrame(self.energy_dict, columns=columns, index=[0])
            df.to_csv("postscf_embedding_energies.csv", header=True, index=None)

    def export_matrices(self):
        """Save all matrices into files."""
        self.scf_wrap.export_matrices()
        for element in self.vemb_dict:
            np.savetxt(element+".txt", self.vemb_dict[element])
