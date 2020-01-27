"""PySCF Utilities for Embedding calculations."""

import numpy as np
import qcelemental as qcel
from pyscf import gto
from pyscf.dft import libxc, gen_grid
from pyscf.dft.numint import eval_ao, eval_rho, eval_mat

from taco.embedding.embpot import EmbPotBase


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


class PyScfEmbPot(EmbPotBase):
    """Base class for embedding potentials.

    Attributes
    ----------
    mol0, mol1 :
        Molecule objects.
    main_info : dict
        Main information to be used in other places:
        atoms, basis
    vemb_dict :  dict
        Container for matrices involved in embedding calculation.
    energy_dict :  dict
        Container for energies involved in embedding calculation.

    Methods
    -------
    __init__(self, frag0_args, frag1_args, emb_args)
    check_molecules(self, mol0, mol1)
    check_emb_arguments(emb_args)
    save_maininfo(self)
    assign_dm(self, nfrag, dm)
    compute_coulomb_potential(self)
    compute_attraction_potential(self)
    compute_nad_potential(self)
    compute_embedding_potential(self)
    export_matrices(self)

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
        self.check_molecules(mol0, mol1)
        self.save_maininfo(mol0)
        EmbPotBase.__init__(self, mol0, mol1, emb_args)

    @staticmethod
    def check_molecules(mol0, mol1):
        """Verify they are PySCF gto.M objects."""
        if not isinstance(mol0, gto.Mole):
            raise TypeError("mol0 must be gto.M PySCF object.")
        if not isinstance(mol1, gto.Mole):
            raise TypeError("mol1 must be gto.M PySCF object.")

    def save_maininfo(self, mol0):
        """Save in a dictionary basic information of mol0 in a simple format."""
        mol0_charges, mol0_coords = get_charges_and_coords(mol0)
        mol0_basis = mol0.basis
        mol0_nbas = mol0.nao_nr()
        self.maininfo = dict(inprog="pyscf", atoms=mol0_charges, basis=mol0_basis,
                             nbas=mol0_nbas)

    def compute_coulomb_potential(self):
        """Compute the electron-electron repulsion potential.

        Returns
        -------
        v_coulomb : np.ndarray(NAO,NAO)
            Coulomb repulsion potential.

        """
        mol0 = self.mol0
        mol1 = self.mol1
        mol1234 = mol1 + mol1 + mol0 + mol0
        shls_slice = (0, mol1.nbas,
                      mol1.nbas, mol1.nbas+mol1.nbas,
                      mol1.nbas+mol1.nbas, mol1.nbas+mol1.nbas+mol0.nbas,
                      mol1.nbas+mol1.nbas+mol0.nbas, mol1234.nbas)
        eris = mol1234.intor('int2e', shls_slice=shls_slice)
        v_coulomb = np.einsum('ab,abcd->cd', self.dm1, eris)
        return v_coulomb

    def compute_attraction_potential(self):
        """Compute the nuclei-electron attraction potentials.

        Returns
        -------
        v0nuc1 : np.ndarray(NAO,NAO)
            Attraction potential between electron density of fragment0
            and nuclei of fragment1.
        v1nuc0 : np.ndarray(NAO,NAO)
            Attraction potential between electron density of fragment0
            and nuclei of fragment1.

        """
        # Nuclear-electron attraction integrals
        mol0_charges, mol0_coords = get_charges_and_coords(self.mol0)
        mol1_charges, mol1_coords = get_charges_and_coords(self.mol1)
        v0_nuc1 = 0
        for i, q in enumerate(mol1_charges):
            self.mol0.set_rinv_origin(mol1_coords[i])
            v0_nuc1 += self.mol0.intor('int1e_rinv') * -q
        v1_nuc0 = 0
        for i, q in enumerate(mol0_charges):
            self.mol1.set_rinv_origin(mol0_coords[i])
            v1_nuc0 += self.mol1.intor('int1e_rinv') * -q
        return v0_nuc1, v1_nuc0

    def compute_nad_potential(self):
        """Compute the non-additive potentials and energies.

        Returns
        -------
        vxc_nad : np.ndarray(NAO,NAO)
            Non-additive Exchange-Correlation + Kinetic potential.

        """
        # Create supersystem
        newatom = '\n'.join([self.mol0.atom, self.mol1.atom])
        system = gto.M(atom=newatom, basis=self.mol0.basis)
        # Construct grid for complex
        grids = gen_grid.Grids(system)
        grids.level = 4
        grids.build()
        ao_mol0 = eval_ao(self.mol0, grids.coords, deriv=0)
        ao_mol1 = eval_ao(self.mol1, grids.coords, deriv=0)
        # Make Complex DM
        ao_both = eval_ao(system, grids.coords, deriv=0)
        nao_mol0 = self.mol0.nao_nr()
        nao_mol1 = self.mol1.nao_nr()
        nao_tot = nao_mol0 + nao_mol1
        dm_both = np.zeros((nao_tot, nao_tot))
        dm_both[:nao_mol0, :nao_mol0] = self.dm0
        dm_both[nao_mol0:, nao_mol0:] = self.dm1
        # Compute DFT non-additive potential and energies
        rho_mol0 = eval_rho(self.mol0, ao_mol0, self.dm0, xctype='LDA')
        rho_mol1 = eval_rho(self.mol1, ao_mol1, self.dm1, xctype='LDA')
        rho_both = eval_rho(system, ao_both, dm_both, xctype='LDA')
        # Compute all densities on a grid
        xc_code = self.emb_args["xc_code"]
        t_code = self.emb_args["t_code"]
        excs, vxcs = get_dft_grid_stuff(xc_code, rho_both, rho_mol0, rho_mol1)
        ets, vts = get_dft_grid_stuff(t_code, rho_both, rho_mol0, rho_mol1)
        vxc_emb = vxcs[0][0] - vxcs[1][0]
        vt_emb = vts[0][0] - vts[1][0]
        # Energy functionals:
        exc_nad = get_nad_energy(grids, excs, rho_both, rho_mol0, rho_mol1)
        et_nad = get_nad_energy(grids, ets, rho_both, rho_mol0, rho_mol1)
        v_nad_xc = eval_mat(self.mol0, ao_mol0, grids.weights, rho_mol0, vxc_emb, xctype='LDA')
        v_nad_t = eval_mat(self.mol0, ao_mol0, grids.weights, rho_mol0, vt_emb, xctype='LDA')
        return (exc_nad, et_nad, v_nad_xc, v_nad_t)

    def compute_embedding_potential(self, dm0=None, dm1=None):
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
        if dm0 is None:
            if self.dm0 is None:
                raise AttributeError("Density matrix for fragment 0 is missing.")
            else:
                dm0 = self.dm0
        else:
            self.assign_dm(0, dm0)
        if dm1 is None:
            if self.dm1 is None:
                raise AttributeError("Density matrix for fragment 1 is missing.")
            else:
                dm1 = self.dm1
        else:
            self.assign_dm(1, dm1)
        # Get DFT non-additive terms
        ref_vnad = self.compute_nad_potential()
        exc_nad, et_nad, v_nad_xc, v_nad_t = ref_vnad
        self.vemb_dict["exc_nad"] = exc_nad
        self.vemb_dict["et_nad"] = et_nad
        # Electrostatic part
        v_coulomb = self.compute_coulomb_potential()
        # Nuclear-electron integrals
        v0_nuc1, v1_nuc0 = self.compute_attraction_potential()
        vemb = v_coulomb + v_nad_xc + v_nad_t + v0_nuc1
        self.vemb_dict["v_coulomb"] = v_coulomb
        self.vemb_dict["v_nad_t"] = v_nad_t
        self.vemb_dict["v_nad_xc"] = v_nad_xc
        self.vemb_dict["v0_nuc1"] = v0_nuc1
        self.vemb_dict["v1_nuc0"] = v1_nuc0
        self.vemb_dict["vemb"] = vemb
        return vemb
