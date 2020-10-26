""" The Base Class for Wrappers. """

import numpy as np

from taco.embedding.emb_pot import EmbPotBase


class PostScfWrap():
    """Base class for Post-SCF from Quantum Chemistry Packages.

    Attributes
    ----------
    emb_pot : EmbPotBase object.
        All information about the embedding potential.
    energy_dict :  dict
        Container for energy results.
    dms_dict :  dict
        Container for matrices involved in embedding calculation.

    Methods
    -------
    __init__(self, scf_wrap)
    check_scf_wrap(self)
    get_density(self)
    format_potential(self, filename)
    save_info(self)
    print_embedding_information(self, to_csv)
    export_matrices(self)
    prepare_for_postscf(self, scfenergy_dict, scfvemb_dict)

    """
    def __init__(self, emb_pot):
        """The wrapper for SCF in QC packages to perform FDET calculations.

        Parameters
        ----------
        emb_args : dict
            Parameters for the embedding calculation:
            method, basis, x_func, c_func, t_func.

        """
        if not isinstance(emb_pot, EmbPotBase):
            raise TypeError("emb_pot must be an instance of the EmbPotBase class.")
        self.emb_pot = emb_pot
        self.energy_dict = {}
        self.dms_dict = {}

    def format_potential(self, filename=None):
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
        if len(self.dms_dict) == 0:
            raise ValueError("No information to be stored, DMs missing.")
        # Get densities from methods
        dm0_final = self.dms_dict["dm0_final"]
        self.emb_pot.assign_dm(0, dm0_final)
        # Get non-additive information
        # Final density functionals
        final_vnad = self.emb_pot.compute_nad_potential()
        self.energy_dict["exc_nad_final"] = final_vnad[0]
        self.energy_dict["et_nad_final"] = final_vnad[1]
        v_nad_xc_final = final_vnad[2]
        v_nad_t_final = final_vnad[3]
        self.energy_dict["int_final_xc"] = np.einsum('ab,ba', v_nad_xc_final, dm0_final)
        self.energy_dict["int_final_t"] = np.einsum('ab,ba', v_nad_t_final, dm0_final)
        # Linearization terms
        if "v_nad_xc_final" not in self.emb_pot.vemb_dict:
            raise KeyError("To save linearization terms the v_nad_xc_final needs to be added to the vemb_dict.")
        if "v_nad_t_final" not in self.emb_pot.vemb_dict:
            raise KeyError("To save linearization terms the v_nad_t_final needs to be added to the vemb_dict.")
        if "int_ref_xc" not in self.emb_pot.vemb_dict:
            raise KeyError("To save linearization terms the int_ref_xc needs to be added to the vemb_dict.")
        if "int_ref_t" not in self.emb_pot.vemb_dict:
            raise KeyError("To save linearization terms the int_ref_t needs to be added to the vemb_dict.")
        int_emb_xc = np.einsum('ab,ba', self.emb_pot.vemb_dict["v_nad_xc_final"], dm0_final)
        int_emb_t = np.einsum('ab,ba', self.emb_pot.vemb_dict["v_nad_t_final"], dm0_final)
        int_ref_xc = self.emb_pot.vemb_dict["int_ref_xc"]
        int_ref_t = self.emb_pot.vemb_dict["int_ref_t"]
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
        for element in self.dms_dict:
            np.savetxt(element+".txt", self.dms_dict[element])

    def prepare_for_postscf(self, scfenergy_dict, scfvemb_dict):
        """Save values vemb_dict for later calculations."""
        self.emb_pot.vemb_dict["v_nad_xc_final"] = scfvemb_dict["v_nad_xc_final"]
        self.emb_pot.vemb_dict["v_nad_t_final"] = scfvemb_dict["v_nad_t_final"]
        self.emb_pot.vemb_dict["int_ref_xc"] = scfenergy_dict["int_ref_xc"]
        self.emb_pot.vemb_dict["int_ref_t"] = scfenergy_dict["int_ref_t"]
