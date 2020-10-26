"""PySCF Utilities for Embedding calculations."""

import numpy as np
from pyscf import gto, lib, df
from pyscf.dft import libxc
from pyscf.dft.numint import eval_ao, eval_rho, eval_mat

from taco.embedding.scf_wrap_single import ScfWrapSingle
from taco.embedding.pyscf_wrap import get_pyscf_method
from taco.embedding.pyscf_emb_pot import get_charges_and_coords
from taco.embedding.cc_gridfns import coulomb_potential_grid, nuclear_attraction_energy


def get_dft_grid_stuff(code, rho_both, rho1, rho2):
    """Evaluate energy densities and potentials on a grid.

    Parameters
    ----------
    code : str
        String with density functional code for PySCF.
    rho_both :  np.ndarray(npoints, dtype=float)
        Total density evaluated on n grid points.
    rho1, rho2 :  np.ndarray(npoints, dtype=float)
        Density of each fragment evaluated on n grid points.

    """
    exc, vxc, fxc, kxc = libxc.eval_xc(code, rho_both)
    exc2, vxc2, fxc2, kxc2 = libxc.eval_xc(code, rho1)
    exc3, vxc3, fxc3, kxc3 = libxc.eval_xc(code, rho2)
    return (exc, exc2, exc3), (vxc, vxc2, vxc3)


def get_coulomb_repulsion(mol, dm, grid):
    """Compute the coulomb repulsion using AO integrals.

    Paramters
    ---------
    mol : gto.M
        Molecule of fragment B
    dm : np.ndarray(NAOs, NAOS)
        Density matrix of fragment B
    grid : np.ndarray((N, 3))
        3D array of coordinates
    """
    v_coul = np.empty_like(grid[:, 0])
    for p0, p1 in lib.prange(0, v_coul.size, 600):
        fakemol = gto.fakemol_for_charges(grid[p0:p1])
        ints = df.incore.aux_e2(mol, fakemol)
        v_coul[p0:p1] = np.einsum('ijp,ij->p', ints, dm)
    return v_coul


def get_electrostatic_potentials(mol0, rho0, dens_func, frag1_charges, grid_args):
    """Compute nuclear attraction between fragments.

    Parameters
    ----------
    mol1, mol2 : gto.M
        Molecule PySCF objects.
    """
    # Construct grid for complex
    ao_mol0 = eval_ao(mol0, grid_args["points"], deriv=0)
    # Evaluate Coulomb repulsion potential
    # Generate other integration grid to evaluate the potential
    points1 = grid_args["points"] + np.array([0.2, 0.2, 0.2])
    rho1 = dens_func(points1)
    v_coul = coulomb_potential_grid(grid_args["points"], points1, grid_args["weights"], rho1)

    # Integrate with mol0 AOs
    v_coulomb = eval_mat(mol0, ao_mol0, grid_args["weights"], rho0, v_coul, xctype='LDA')

    # Nuclear-electron attraction integrals
    mol0_charges, mol0_coords = get_charges_and_coords(mol0)
    mol1_charges = frag1_charges["charges"]
    mol1_coords = frag1_charges["charges_coords"]
    v0_nuc1 = 0
    for i, q in enumerate(mol1_charges):
        mol0.set_rinv_origin(mol1_coords[i])
        v0_nuc1 += mol0.intor('int1e_rinv') * -q
    # Create dictionary
    elst_potentials = dict(v_coulomb=v_coulomb, v0_nuc1=v0_nuc1)
    return elst_potentials


def get_density_from_dm(mol, dm, points, xctype='LDA', deriv=0):
    """Compute density on a grid from the density matrix.

    Parameters
    ----------
    mol : gto.M
        Molecule PySCF object.
    dm : np.ndarray
        Density matrix corresponding to mol.
    points : np.ndarray
        Grid points where the density is evaluated.
    xctype : str
        Type of functional it's used later. This defines
        how many derivatives or rho will be computed.
    deriv : int
        Number of derivatives needed for orbitals and density.

    Returns
    -------
    rho : np.ndarray(npoints, dtype=float)
        Density on a grid.

    """
    ao_mol = eval_ao(mol, points, deriv=deriv)
    rho = eval_rho(mol, ao_mol, dm, xctype=xctype)
    return rho


def get_nad_energy(grid_weights, energies, rho_both, rho0, rho1):
    """Calculate non-additive energy.

    Parameters
    ----------
    grid_weigths : np.ndarray
        Integration grid weights.
    energies : list
        List of individual energies: total, [fragment1, fragment2]
    rho_both :  np.ndarray(npoints, dtype=float)
        Total density evaluated on n grid points.
    rho0, rho1 :  np.ndarray(npoints, dtype=float)
        Density of each fragment evaluated on n grid points.

    """
    e_nad = np.dot(rho_both*grid_weights, energies[0])
    e_nad -= np.dot(rho0*grid_weights, energies[1])
    e_nad -= np.dot(rho1*grid_weights, energies[2])
    return e_nad


def compute_nad_terms(mol0, rho0, rho1, grid_args, emb_args):
    """Compute the non-additive potentials and energies.

    Parameters
    ----------
    mol0 : PySCF gto.M
        Molecule objects.
    emb_args : dict
        Information of embedding calculation.

    """
    # Construct grid for complex
    ao_mol0 = eval_ao(mol0, grid_args["points"], deriv=0)

    # Make Complex DM
    rho_both = rho0 + rho1

    # Compute all densities on a grid
    xc_code = emb_args["xc_code"]
    t_code = emb_args["t_code"]
    excs, vxcs = get_dft_grid_stuff(xc_code, rho_both, rho0, rho1)
    ets, vts = get_dft_grid_stuff(t_code, rho_both, rho0, rho1)
    vxc_emb = vxcs[0][0] - vxcs[1][0]
    vt_emb = vts[0][0] - vts[1][0]
    # Energy functionals:
    exc_nad = get_nad_energy(grid_args["weights"], excs, rho_both, rho0, rho1)
    et_nad = get_nad_energy(grid_args["weights"], ets, rho_both, rho0, rho1)

    v_nad_xc = eval_mat(mol0, ao_mol0, grid_args["weights"], rho0, vxc_emb, xctype='LDA')
    v_nad_t = eval_mat(mol0, ao_mol0, grid_args["weights"], rho0, vt_emb, xctype='LDA')
    return (exc_nad, et_nad, v_nad_xc, v_nad_t)


def compute_nuclear_repulsion(charges1, coords1, charges2, coords2):
    """Compute nuclear repulsion between two fragments.

    Parameters
    ----------
    charges1, charges2 :  np.ndarray
        Charges of fragments.
    coords1, coords1 :  np.ndarray
        Position on space of these charges.

    """
    result = 0
    for i, q1 in enumerate(charges1):
        for j, q2 in enumerate(charges2):
            d = np.linalg.norm(coords1[i]-coords2[j])
            result += q1*q2/d
    return result


class PyScfWrapSingle(ScfWrapSingle):
    """PySCF wrapper for embedding calculations.

    Attributes
    ----------
    energy_dict :  dict
        Container for energy results.
    vemb_dict :  dict
        Container for matrices involved in embedding calculation.
    mol :
        Molecule objects.
    method :
        Method objects.

    Methods
    -------
    __init__(self, frag0_args, dens_func, emb_args)
    create_fragment(self, frag_args)
    compute_embedding_potential(self)
    run_embedding(self)
    save_info(self)
    print_embedding_information(self, to_csv)
    export_matrices(self)

    """
    def __init__(self, frag0_args, frag1_charges, dens1_func, grid_args, emb_args):
        """Wrapper for PySCF methods.

        Parameters
        ----------
        frag0_args : dict
            Parameters for fragment 0:
            molecule, method, basis, xc_code, etc.
        frag1_charges : dict
            Charges for fragment 1:
            Nuclear/effective charges, charges_coords (both np.ndarray).
        dens1_func :  callable
            A function to evaluate the density on a given grid.
        emb_args : dict
            Parameters for the embedding calculation:
            method, basis, x_func, c_func, t_func.

        """
        ScfWrapSingle.__init__(self, frag0_args, frag1_charges, dens1_func, grid_args, emb_args)
        self.create_fragment(frag0_args)
        self.check_charge_arguments(frag1_charges)
        self.check_grid_arguments(grid_args)
        self.grid_args = grid_args
        self.check_emb_arguments(emb_args)
        self.emb_method = get_pyscf_method(emb_args)

    def create_fragment(self, frag_args):
        """Save fragment information.

        Parameters
        ----------
        frag_args : dict
            Parameters for individual fragments:
            molecule, method, basis.
        """
        self.check_qc_arguments(frag_args)
        self.method = get_pyscf_method(frag_args)
        self.mol = self.method.mol_pyscf

    def compute_embedding_potential(self):
        """Compute embedding potential.

        Returns
        -------
        vemb : np.ndarray
            Embedding potential as a Fock-like matrix.

        """
        # First get the density matrices
        dm0 = self.method.get_density()
        rho0 = get_density_from_dm(self.mol, dm0, self.grid_args["points"])
        rho1 = self.dens1_func(self.grid_args["points"])

        # Get DFT non-additive terms
        ref_vnad = compute_nad_terms(self.mol, rho0, rho1, self.grid_args, self.emb_args)
        exc_nad, et_nad, v_nad_xc, v_nad_t = ref_vnad
        self.energy_dict["exc_nad"] = exc_nad
        self.energy_dict["et_nad"] = et_nad
        # Electrostatic part
        elst_potentials = get_electrostatic_potentials(self.mol, rho0, self.dens1_func,
                                                       self.frag1_charges, self.grid_args)
        v_coulomb = elst_potentials["v_coulomb"]
        v0_nuc1 = elst_potentials["v0_nuc1"]
        # Nuclear-electron integrals
        vemb = v_coulomb + v_nad_xc + v_nad_t + v0_nuc1
        self.vemb_dict["v_coulomb"] = v_coulomb
        self.vemb_dict["v_nad_t"] = v_nad_t
        self.vemb_dict["v_nad_xc"] = v_nad_xc
        self.vemb_dict["v0_nuc1"] = v0_nuc1
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

    def save_info(self, get_all=True):
        """Save information after embedding calculation.

        Parameters:
        get_all : bool
            If all terms must be computed and saved.
            The default is true, so the nuclear-electron repulsion between frag0 and
            the density of frag1 is computed.


        """
        # Get densities from methods
        dm0 = self.method0.get_density()
        dm0_final = self.emb_method.get_density()
        self.vemb_dict["dm0_ref"] = dm0
        self.vemb_dict["dm0_final"] = dm0_final

        # Get electrostatics
        charges0, coords0 = get_charges_and_coords(self.mol)
        charges1 = self.frag1_charges["charges"]
        coords1 = self.frag1_charges["charges_coords"]
        rho1 = self.dens1_func(self.grid_args["points"])
        rho0_final = get_density_from_dm(self.mol, dm0_final, self.grid_args["points"])
        self.energy_dict["nuc0_nuc1"] = compute_nuclear_repulsion(charges0, coords0, charges1, coords1)
        self.energy_dict["rho0_rho1"] = np.einsum('ab,ba', self.vemb_dict["v_coulomb"], dm0_final)
        self.energy_dict["nuc0_rho1"] = np.einsum('ab,ba', self.vemb_dict["v0_nuc1"], dm0_final)
        if get_all:
            v1_nuc0 = nuclear_attraction_energy(self.frag1_charges["charges"],
                                                self.frag1_charges["charges_coords"],
                                                self.grid_args["points"],
                                                self.grid_args["weights"], rho0_final)
            self.energy_dict["nuc1_rho0"] = v1_nuc0
        # Get non-additive information
        # Final density functionals
        final_vnad = compute_nad_terms(self.mol, rho0_final, rho1, self.grid_args, self.emb_args)
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
