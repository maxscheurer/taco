"""PySCF Utilities for Embedding calculations."""

import numpy as np
import qcelemental as qcel
from pyscf import gto
from pyscf.dft import libxc, gen_grid
from pyscf.dft.numint import eval_ao, eval_rho, eval_mat

from taco.embedding.scf_wrap import ScfWrap
from taco.methods.scf_pyscf import ScfPyScf


def get_charges_and_coords(mol):
    """Return arrays with charges and coordinates."""
    bohr2a = qcel.constants.conversion_factor("bohr", "angstrom")
    coords = []
    charges = []
    atm_str = mol.atom.split()
    for i in range(mol.natm):
        tmp = [float(f)/bohr2a for f in atm_str[i*4+1:(i*4)+4]]
        coords.append(tmp)
        charges.append(mol._atm[i][0])
    coords = np.array(coords)
    charges = np.array(charges, dtype=int)
    return charges, coords


def get_coulomb(mol1, mol2, dm2):
    """Compute Coulomb repulsion between fragments.

    Parameters
    ----------
    mol1, mol2 : gto.M
        Molecule PySCF objects.
    dm2 : np.ndarray
        Density matrix of fragment 2.
    """
    mol1234 = mol2 + mol2 + mol1 + mol1
    shls_slice = (0, mol2.nbas,
                  mol2.nbas, mol2.nbas+mol2.nbas,
                  mol2.nbas+mol2.nbas, mol2.nbas+mol2.nbas+mol1.nbas,
                  mol2.nbas+mol2.nbas+mol1.nbas, mol1234.nbas)
    eris = mol1234.intor('int2e', shls_slice=shls_slice)
    v_coulomb = np.einsum('ab,abcd->cd', dm2, eris)
    return v_coulomb


def get_attraction_potential(mol1, mol2):
    """Compute nuclear attraction between fragments.

    Parameters
    ----------
    mol1, mol2 : gto.M
        Molecule PySCF objects.
    """
    # Nuclear-electron attraction integrals
    mol1_charges, mol1_coords = get_charges_and_coords(mol1)
    mol2_charges, mol2_coords = get_charges_and_coords(mol2)
    vAnucB = 0
    for i, q in enumerate(mol2_charges):
        mol1.set_rinv_origin(mol2_coords[i])
        vAnucB += mol1.intor('int1e_rinv') * -q

    vBnucA = 0
    for i, q in enumerate(mol1_charges):
        mol2.set_rinv_origin(mol1_coords[i])
        vBnucA += mol2.intor('int1e_rinv') * -q
    return vAnucB, vBnucA


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


def get_nad_energy(grid, energies, rho_both, rho1, rho2):
    """Calculate non-additive energy.

    Parameters
    ----------
    grid : len_grids.grids
        Integration grid object.
    energies : list
        List of individual energies: total, [fragment1, fragment2]
    rho_both :  np.ndarray(npoints, dtype=float)
        Total density evaluated on n grid points.
    rho1, rho2 :  np.ndarray(npoints, dtype=float)
        Density of each fragment evaluated on n grid points.

    """
    e_nad = np.dot(rho_both*grid.weights, energies[0])
    e_nad -= np.dot(rho1*grid.weights, energies[1])
    e_nad -= np.dot(rho2*grid.weights, energies[2])
    return e_nad


def compute_nad_terms(mol0, mol1, dm0, dm1, emb_args):
    """Compute the non-additive potentials and energies.

    Parameters
    ----------
    mol : PySCF gto.M
        Molecule objects.
    emb_args : dict
        Information of embedding calculation.

    """
    # Create supersystem
    newatom = '\n'.join([mol0.atom, mol1.atom])
    system = gto.M(atom=newatom, basis=mol0.basis)
    # Construct grid for complex
    grids = gen_grid.Grids(system)
    grids.level = 4
    grids.build()
    ao_mol0 = eval_ao(mol0, grids.coords, deriv=0)
    ao_mol1 = eval_ao(mol1, grids.coords, deriv=0)
    # Make Complex DM
    ao_both = eval_ao(system, grids.coords, deriv=0)
    nao_mol0 = mol0.nao_nr()
    nao_mol1 = mol1.nao_nr()
    nao_tot = nao_mol0 + nao_mol1
    dm_both = np.zeros((nao_tot, nao_tot))

    dm_both[:nao_mol0, :nao_mol0] = dm0
    dm_both[nao_mol0:, nao_mol0:] = dm1

    # Compute DFT non-additive potential and energies
    rho_mol0 = eval_rho(mol0, ao_mol0, dm0, xctype='LDA')
    rho_mol1 = eval_rho(mol1, ao_mol1, dm1, xctype='LDA')
    rho_both = eval_rho(system, ao_both, dm_both, xctype='LDA')
    # Compute all densities on a grid
    xc_code = emb_args["xc_code"]
    t_code = emb_args["t_code"]
    excs, vxcs = get_dft_grid_stuff(xc_code, rho_both, rho_mol0, rho_mol1)
    ets, vts = get_dft_grid_stuff(t_code, rho_both, rho_mol0, rho_mol1)
    vxc_emb = vxcs[0][0] - vxcs[1][0]
    vt_emb = vts[0][0] - vts[1][0]
    # Energy functionals:
    exc_nad = get_nad_energy(grids, excs, rho_both, rho_mol0, rho_mol1)
    et_nad = get_nad_energy(grids, ets, rho_both, rho_mol0, rho_mol1)

    v_nad_xc = eval_mat(mol0, ao_mol0, grids.weights, rho_mol0, vxc_emb, xctype='LDA')
    v_nad_t = eval_mat(mol0, ao_mol0, grids.weights, rho_mol0, vt_emb, xctype='LDA')
    return (exc_nad, et_nad, v_nad_xc, v_nad_t)


def compute_nuclear_repulsion(mol1, mol2):
    """Compute nuclear repulsion between two fragments.

    Parameters
    ----------
    mol1, mol2 : gto.M
        Molecule objects

    """
    result = 0
    charges1, coord1 = get_charges_and_coords(mol1)
    charges2, coord2 = get_charges_and_coords(mol2)
    for i, q1 in enumerate(charges1):
        for j, q2 in enumerate(charges2):
            d = np.linalg.norm(coord1[i]-coord2[j])
            result += q1*q2/d
    return result


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
        ref_vnad = compute_nad_terms(self.mol0, self.mol1, dm0, dm1, self.emb_args)
        exc_nad, et_nad, v_nad_xc, v_nad_t = ref_vnad
        self.energy_dict["exc_nad"] = exc_nad
        self.energy_dict["et_nad"] = et_nad
        # Electrostatic part
        v_coulomb = get_coulomb(self.mol0, self.mol1, dm1)
        # Nuclear-electron integrals
        v0_nuc1, v1_nuc0 = get_attraction_potential(self.mol0, self.mol1)
        vemb = v_coulomb + v_nad_xc + v_nad_t + v0_nuc1
        self.vemb_dict["v_coulomb"] = v_coulomb
        self.vemb_dict["v_nad_t"] = v_nad_t
        self.vemb_dict["v_nad_xc"] = v_nad_xc
        self.vemb_dict["v0_nuc1"] = v0_nuc1
        self.vemb_dict["v1_nuc0"] = v1_nuc0
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
        final_vnad = compute_nad_terms(self.emb_method.mol_pyscf, self.mol1, dm0_final,
                                       dm1, self.emb_args)
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
