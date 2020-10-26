"""PySCF Utilities for Embedding calculations."""

import numpy as np
from taco.embedding.scf_wrap import ScfWrap
from taco.embedding.pyscf_emb_pot import PyScfEmbPot, compute_nuclear_repulsion
from taco.methods.scf_pyscf import ScfPyScf


def get_pyscf_method(args):
    """Create PySCF method object.
    Parameters
    ----------
    args : dict
        Parameters to initialize a method object. It must contain:
        mol : qcelement.models.Molecule
            Molecule object.
        method : str
            Method name. Available options at the moment:
            HF or DFT.
        basis : str
            Basis set name. Any of the PySCF basis sets.
        xc_code : str
            Density functional code, only needed for DFT methods.
    """
    return ScfPyScf(args['mol'], args['basis'], args['method'], args['xc_code'])


class PyScfWrap(ScfWrap):
    """PySCF wrapper for embedding calculations.

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
    pot_object : EmbPot object
        Embedding potential constructor.

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
        """Wrapper for PySCF methods.

        Parameters
        ----------
        frag_args : dict
            Parameters for individual fragments:
            molecule, method, basis, xc_code, etc.
        emb_args : dict
            Parameters for the embedding calculation:
            method, basis, x_func, c_func, t_func.

        """
        ScfWrap.__init__(self, frag0_args, frag1_args, emb_args)
        self.create_fragments(frag0_args, frag1_args)
        self.check_emb_arguments(emb_args)
        self.emb_method = get_pyscf_method(emb_args)
        self.pot_object = None

    def create_fragments(self, frag0_args, frag1_args):
        """Save fragment information.

        Parameters
        ----------
        frag_args : dict
            Parameters for individual fragments:
            molecule, method, basis.
        """
        self.check_basic_arguments(frag0_args)
        self.check_basic_arguments(frag1_args)
        self.method0 = get_pyscf_method(frag0_args)
        self.method1 = get_pyscf_method(frag1_args)
        self.mol0 = self.method0.mol_pyscf
        self.mol1 = self.method1.mol_pyscf

    def compute_embedding_potential(self):
        """Compute embedding potential.

        Returns
        -------
        vemb : np.ndarray
            Embedding potential as a Fock-like matrix.

        """
        # First get the density matrices
        dm0 = self.method0.get_density()
        dm1 = self.method1.get_density()
        # Get DFT non-additive terms
        self.pot_object = PyScfEmbPot(self.mol0, self.mol1, self.emb_args)
        vemb = self.pot_object.compute_embedding_potential(dm0, dm1)
        # Copy info into wrap dictionaries
        for key in self.pot_object.vemb_dict:
            if key.startswith("v"):
                self.vemb_dict[key] = self.pot_object.vemb_dict[key]
            elif key.startswith("e"):
                self.energy_dict[key] = self.pot_object.vemb_dict[key]
        return vemb

    def run_embedding(self):
        """Run FDET embedding calculation."""
        vemb = self.compute_embedding_potential()
        # Add embedding potential to Fock matrix and run SCF
        self.emb_method.perturb_fock(vemb)
        # TODO: pass convergence tolerance from outside
        # TODO: conv_tol_grad missing
        self.emb_method.solve_scf(conv_tol=1e-14)
        # Save final values
        self.save_info()

    def save_info(self):
        """Save information after embedding calculation."""
        # Get densities from methods
        dm0 = self.method0.get_density()
        dm1 = self.method1.get_density()
        dm0_final = self.emb_method.get_density()
        self.vemb_dict["dm0_ref"] = dm0
        self.vemb_dict["dm1_ref"] = dm1
        self.vemb_dict["dm0_final"] = dm0_final

        # Get electrostatics
        self.energy_dict["rho0_rho1"] = np.einsum('ab,ba', self.vemb_dict["v_coulomb"], dm0_final)
        self.energy_dict["nuc0_rho1"] = np.einsum('ab,ba', self.vemb_dict["v0_nuc1"], dm0_final)
        self.energy_dict["nuc1_rho0"] = np.einsum('ab,ba', self.vemb_dict["v1_nuc0"], dm1)
        self.energy_dict["nuc0_nuc1"] = compute_nuclear_repulsion(self.mol0, self.mol1)
        # Get non-additive information
        # Final density functionals
        self.pot_object.assign_dm(0, dm0_final)
        final_vnad = self.pot_object.compute_nad_potential()
        self.energy_dict["exc_nad_final"] = final_vnad[0]
        self.energy_dict["et_nad_final"] = final_vnad[1]
        self.vemb_dict["v_nad_xc_final"] = final_vnad[2]
        self.vemb_dict["v_nad_t_final"] = final_vnad[3]
        int_ref_xc = np.einsum('ab,ba', self.vemb_dict["v_nad_xc"], dm0)
        int_ref_t = np.einsum('ab,ba', self.vemb_dict["v_nad_t"], dm0)
        self.energy_dict["int_final_xc"] = np.einsum('ab,ba', self.vemb_dict["v_nad_xc_final"], dm0)
        self.energy_dict["int_final_t"] = np.einsum('ab,ba', self.vemb_dict["v_nad_t_final"], dm0)
        # Linearization terms
        int_emb_xc = np.einsum('ab,ba', self.vemb_dict["v_nad_xc"], dm0_final)
        int_emb_t = np.einsum('ab,ba', self.vemb_dict["v_nad_t"], dm0_final)
        self.energy_dict["int_ref_xc"] = int_ref_xc
        self.energy_dict["int_ref_t"] = int_ref_t
        self.energy_dict["int_emb_xc"] = int_emb_xc
        self.energy_dict["int_emb_t"] = int_emb_t
        self.energy_dict["deltalin"] = (int_emb_xc - int_ref_xc) + (int_emb_t - int_ref_t)
