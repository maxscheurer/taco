
import numpy as np
from scipy.linalg import block_diag

import psi4


def run_co_h2o_psi4(basis):
    def build_supersystem(mol1, mol2):
        # geom, mass, elem, elez, uniq
        geom1, _, elem1, _, _ = mol1.to_arrays()
        geom2, _, elem2, _, _ = mol2.to_arrays()
        geom_merged = np.vstack((geom1, geom2))
        elem_merged = np.append(elem1, elem2).reshape(geom_merged.shape[0], 1)
        mol = np.hstack((elem_merged, geom_merged))
        mol_string = ""
        for a in mol:
            astr = " ".join([str(en) for en in a])
            mol_string += astr + "\n"
        temp = """
        0 1
        {mol}
        units au
        symmetry c1
        no_reorient
        no_com
        """.format(mol=mol_string)
        return psi4.geometry(temp)

    h2o = psi4.geometry("""
    0 1
    O  -7.9563726699    1.4854060709    0.1167920007
    H  -6.9923165534    1.4211335985    0.1774706091
    H  -8.1058463545    2.4422204631    0.1115993752
    symmetry c1
    no_reorient
    no_com
    """)

    co = psi4.geometry("""
    0 1
    C  -3.6180905689    1.3768035675   -0.0207958979
    O  -4.7356838533    1.5255563000    0.1150239130
    symmetry c1
    no_reorient
    no_com
    """)

    psi4.core.be_quiet()

    psi4.set_options({'basis': basis})
    e_a, wfn_a = psi4.energy("SCF", molecule=co, return_wfn=True, e_convergence=1e-12)
    bas_a = wfn_a.basisset()
    psi4.core.clean()

    psi4.set_options({'basis': basis})
    e_b, wfn_b = psi4.energy("SCF", molecule=h2o, return_wfn=True, e_convergence=1e-12)
    dm_tot_b = wfn_b.Da().np + wfn_b.Db().np
    bas_b = wfn_b.basisset()
    psi4.core.clean()

    # Electronic repulsion
    mints = psi4.core.MintsHelper(wfn_a)
    eri = mints.ao_eri(bas_b, bas_b, bas_a, bas_a)
    v_j = np.einsum('ab,abcd->cd', dm_tot_b, eri)

    # Nuclear-electron attraction
    def compute_nucpot(molb, bas_a):
        """Coompute the nuclear potential between nuclei of B and rhoA.

        Parameters
        ----------
        molb :
            Molecule of B
        bas_a : int
            Basis set of molecule A

        Returns
        -------
        nuc_potential_b : np.ndarray(dtype=float)
            The nuclear attraction potential

        """
        nuc_potential_b = psi4.core.ExternalPotential()
        for i in range(molb.natom()):
            geom = np.array([molb.x(i), molb.y(i), molb.z(i)])
            if molb.units() == 'Angstrom':
                geom *= psi4.constants.bohr2angstroms
            nuc_potential_b.addCharge(molb.Z(i), *geom)
        v_b = nuc_potential_b.computePotentialMatrix(bas_a)
        return v_b

    vAnucB = compute_nucpot(h2o, bas_a)
    vBnucA = compute_nucpot(co, bas_b)

    # DFT nad potential
    def compute_kinetic_tf(rho):
        """Thomas-Fermi kinetic energy functional."""
        cf = 2.8712
        et = cf*np.power(rho, 5./3.)
        vt = cf*5/3*np.power(rho, 2./3.)
        return et, vt

    system = build_supersystem(co, h2o)
    basis_obj = psi4.core.BasisSet.build(system, 'ORBITAL', basis)

    grid = psi4.core.DFTGrid.build(system, basis_obj)

    slater = psi4.core.LibXCFunctional("XC_LDA_X", True)
    vwn = psi4.core.LibXCFunctional("XC_LDA_C_VWN", True)

    superfunc = psi4.core.SuperFunctional()
    superfunc.add_x_functional(slater)
    superfunc.add_c_functional(vwn)
    npoints = psi4.core.get_option("SCF", "DFT_BLOCK_MAX_POINTS")
    superfunc.set_max_points(npoints)
    superfunc.set_deriv(1)
    superfunc.allocate()

    # Contruct potentials
    Vpot = psi4.core.VBase.build(basis_obj, superfunc, "RV")
    Vpot.initialize()
    Da = wfn_a.Da()
    Db = wfn_b.Db()
    dma = Da.np
    dmb = Db.np
    len_A = len(dma)
    dazeros = np.zeros_like(dma)
    dbzeros = np.zeros_like(dmb)
    dm_both = block_diag(dma, dmb)
    dma_sup = block_diag(dma, dbzeros)
    dmb_sup = block_diag(dazeros, dmb)
    Dab = psi4.core.Matrix.from_array(dm_both)
    # Set density matrix
    Vpot.set_D([Dab])
    # Check that went well
    points_func = Vpot.properties()[0]
    points_func.set_pointers(Dab)

    nelec_a = 0
    nelec_b = 0
    nelec_tot = 0
    exc_a = 0
    exc_b = 0
    exc_tot = 0
    et_a = 0
    et_b = 0
    et_tot = 0
    V = np.zeros_like(dm_both)
    Vt = np.zeros_like(dm_both)
    for b in range(len(grid.blocks())):
        # Obtain block information
        block = grid.blocks()[b]
        points_func.compute_points(block)
        npoints = block.npoints()
        lpos = np.array(block.functions_local_to_global())
        # Obtain the grid weight
        w = np.array(block.w())

        # Compute phi!
        phi = np.array(points_func.basis_values()["PHI"])[:npoints, :lpos.shape[0]]

        # Build a local slice of Densities
        lD = dm_both[(lpos[:, None], lpos)]
        lDa = dma_sup[(lpos[:, None], lpos)]
        lDb = dmb_sup[(lpos[:, None], lpos)]

        # Copmute rho
        rho = 2.0 * np.einsum('pm,mn,pn->p', phi, lD, phi)
        rhoa = 2.0 * np.einsum('pm,mn,pn->p', phi, lDa, phi)
        rhob = 2.0 * np.einsum('pm,mn,pn->p', phi, lDb, phi)

        inp_ab = {}
        inp_ab["RHO_A"] = psi4.core.Vector.from_array(rho)
        inpa = {}
        inpa["RHO_A"] = psi4.core.Vector.from_array(rhoa)
        inpb = {}
        inpb["RHO_A"] = psi4.core.Vector.from_array(rhob)

        # Compute the AB functional part
        ret = superfunc.compute_functional(inp_ab, -1)
        vk_tot = np.array(ret["V"])[:npoints]
        et, vt = compute_kinetic_tf(rho)
        et_tot += np.einsum('a,a->', w, et)
        exc_tot += np.einsum('a,a->', w, vk_tot)
        v_rho_tot = np.array(ret["V_RHO_A"])[:npoints]
        vt_tot = vt

        # Compute the A functional part
        reta = superfunc.compute_functional(inpa, -1)
        vk_a = np.array(ret["V"])[:npoints]
        et, vt = compute_kinetic_tf(rhoa)
        et_a += np.einsum('a,a->', w, et)
        exc_a += np.einsum('a,a->', w, vk_a)
        v_rho_a = np.array(reta["V_RHO_A"])[:npoints]
        vt_tot -= vt

        # Compute the B functional part
        retb = superfunc.compute_functional(inpb, -1)
        vk_b = np.array(retb["V"])[:npoints]
        et, vt = compute_kinetic_tf(rhob)
        et_b += np.einsum('a,a->', w, et)
        exc_b += np.einsum('a,a->', w, vk_b)

        # Do B energy for final evaluation

        # This is fine
        nelec_tot += np.einsum('a,a->', w, rho)
        nelec_a += np.einsum('a,a->', w, rhoa)
        nelec_b += np.einsum('a,a->', w, rhob)

        # Compute the XC derivative.
        v_rho_tot -= v_rho_a
        Vtmp = np.einsum('pb,p,p,pa->ab', phi, v_rho_tot, w, phi)
        Vtmp2 = np.einsum('pb,p,p,pa->ab', phi, vt_tot, w, phi)

        # Add the temporary back to the larger array by indexing, ensure it is symmetric
        V[(lpos[:, None], lpos)] += 0.5 * (Vtmp + Vtmp.T)
        Vt[(lpos[:, None], lpos)] += 0.5 * (Vtmp2 + Vtmp2.T)

    # initialize dictionary
    embdic = {}
    # Non-electrostatic, non-additive contributions
    vxc_nad = V[:len_A, :len_A]
    vt_nad = Vt[:len_A, :len_A]
    int_ref_xc = 2*np.einsum('ab,ba', vxc_nad, dma)
    int_ref_t = 2*np.einsum('ab,ba', vt_nad, dma)
    embdic['exc_nad'] = exc_tot - exc_a - exc_b
    embdic['et_nad'] = et_tot - et_a - et_b

    # Re-evaluate HF
    base_wfn = psi4.core.Wavefunction.build(co)
    # Embedding potential
    extra_op = psi4.core.Matrix.from_array(vxc_nad + vt_nad + v_j + vAnucB)
    scf_wfn = psi4.driver.proc.scf_wavefunction_factory("HF", base_wfn, "RHF")
    scf_wfn.initialize()
    # Add the operator (psi4 matrix) to the core Hamiltonian matrix
    scf_wfn.H().add(extra_op)
    scf_wfn.e_convergence = 1e-12
    scf_wfn.iterations()
    scf_wfn.finalize_energy()
    dma_final = scf_wfn.Da().np

    # Use final DM of A and evaluate energy terms
    embdic['rhoArhoB'] = 2*np.einsum('ab,ba', v_j, dma_final)
    embdic['nucArhoB'] = 2*np.einsum('ab,ba', vAnucB, dma_final)
    embdic['nucBrhoA'] = 2*np.einsum('ab,ba', vBnucA, dmb)
    # Linearization terms
    int_emb_xc = 2*np.einsum('ab,ba', vxc_nad, dma_final)
    int_emb_t = 2*np.einsum('ab,ba', vt_nad, dma_final)
    embdic['int_ref_xc'] = int_ref_xc
    embdic['int_ref_t'] = int_ref_t
    embdic['int_emb_xc'] = int_emb_xc
    embdic['int_emb_t'] = int_emb_t
    embdic['deltalin'] = (int_emb_xc - int_ref_xc) + (int_emb_t - int_ref_t)
    psi4.core.clean()
    return embdic


def run_co_h2o_psi4_sto3g():
    # Get HF-in-HF embedding energies
    embdic = run_co_h2o_psi4('sto-3g')
    # Electronstatic terms
    # TODO: check why difference is not smaller
    qchem_rho_A_rho_B = 20.9457553682
    qchem_rho_A_Nuc_B = -21.1298173325
    qchem_rho_B_Nuc_A = -20.8957755874
    assert abs(qchem_rho_A_rho_B - embdic['rhoArhoB']) < 1e-4
    assert abs(qchem_rho_A_Nuc_B - embdic['nucArhoB']) < 1e-5
    assert abs(qchem_rho_B_Nuc_A - embdic['nucBrhoA']) < 1e-4
    # DFT related terms
    qchem_int_ref_xc = -0.0011361532
    qchem_int_ref_t = 0.0022364179
    qchem_exc_nad = -0.0021105605
    qchem_et_nad = 0.0030018734
    qchem_int_emb_xc = -0.0011379466
    qchem_int_emb_t = 0.0022398242
    qchem_deltalin = 0.0000016129
    assert abs(qchem_et_nad - embdic['et_nad']) < 1e-6
    assert abs(qchem_exc_nad - embdic['exc_nad']) < 1e-6
    assert abs(qchem_int_ref_t - embdic['int_ref_t']) < 1e-6
    assert abs(qchem_int_ref_xc - embdic['int_ref_xc']) < 1e-6
    assert abs(qchem_int_emb_t - embdic['int_emb_t']) < 1e-6
    assert abs(qchem_int_emb_xc - embdic['int_emb_xc']) < 1e-6
    assert abs(qchem_deltalin - embdic['deltalin']) < 1e-8


def run_co_h2o_psi4_dz():
    # Get HF-in-HF embedding energies
    embdic = run_co_h2o_psi4('cc-pvdz')
    # Electronstatic terms
    # TODO: check why difference is not smaller
    qchem_rho_A_rho_B = 20.9054911884
    qchem_rho_A_Nuc_B = -21.1510526049
    qchem_rho_B_Nuc_A = -20.8349849585
    assert abs(qchem_rho_A_rho_B - embdic['rhoArhoB']) < 1e-4
    assert abs(qchem_rho_A_Nuc_B - embdic['nucArhoB']) < 1e-5
    assert abs(qchem_rho_B_Nuc_A - embdic['nucBrhoA']) < 1e-4
    # DFT related terms
    qchem_int_ref_xc = -0.0015514201
    qchem_int_ref_t = 0.0025414161
    qchem_exc_nad = -0.0025622100
    qchem_et_nad = 0.003191753
    qchem_int_emb_xc = -0.0016014043
    qchem_int_emb_t = 0.0026148569
    qchem_deltalin = 0.0000234566
    assert abs(qchem_et_nad - embdic['et_nad']) < 1e-6
    assert abs(qchem_exc_nad - embdic['exc_nad']) < 1e-6
    assert abs(qchem_int_ref_t - embdic['int_ref_t']) < 1e-6
    assert abs(qchem_int_ref_xc - embdic['int_ref_xc']) < 1e-6
    assert abs(qchem_int_emb_t - embdic['int_emb_t']) < 1e-6
    assert abs(qchem_int_emb_xc - embdic['int_emb_xc']) < 1e-6
    assert abs(qchem_deltalin - embdic['deltalin']) < 1e-6


def run_co_h2o_psi4_tz():
    # Get HF-in-HF embedding energies
    embdic = run_co_h2o_psi4('cc-pvtz')
    # Electronstatic terms
    # TODO: check why difference is not smaller
    qchem_rho_A_rho_B = 20.9042017461
    qchem_rho_A_Nuc_B = -21.1526053452
    qchem_rho_B_Nuc_A = -20.8322820052
    assert abs(qchem_rho_A_rho_B - embdic['rhoArhoB']) < 1e-5
    assert abs(qchem_rho_A_Nuc_B - embdic['nucArhoB']) < 1e-5
    assert abs(qchem_rho_B_Nuc_A - embdic['nucBrhoA']) < 1e-5
    # DFT related terms
    qchem_int_ref_xc = -0.0018569491
    qchem_int_ref_t = 0.0029461194
    qchem_exc_nad = -0.0029467362
    qchem_et_nad = 0.0036240425
    qchem_int_emb_xc = -0.0019562173
    qchem_int_emb_t = 0.0030885771
    qchem_deltalin = 0.0000431896
    assert abs(qchem_et_nad - embdic['et_nad']) < 1e-6
    assert abs(qchem_exc_nad - embdic['exc_nad']) < 1e-6
    assert abs(qchem_int_ref_t - embdic['int_ref_t']) < 1e-6
    assert abs(qchem_int_ref_xc - embdic['int_ref_xc']) < 1e-6
    assert abs(qchem_int_emb_t - embdic['int_emb_t']) < 1e-6
    assert abs(qchem_int_emb_xc - embdic['int_emb_xc']) < 1e-6
    assert abs(qchem_deltalin - embdic['deltalin']) < 1e-7


def run_co_h2o_psi4_qz():
    # Get HF-in-HF embedding energies
    embdic = run_co_h2o_psi4('cc-pvqz')
    # Electronstatic terms
    # TODO: check why difference is not smaller
    qchem_rho_A_rho_B = 20.9070590943
    qchem_rho_A_Nuc_B = -21.1558918145
    qchem_rho_B_Nuc_A = -20.8320256245
    assert abs(qchem_rho_A_rho_B - embdic['rhoArhoB']) < 1e-5
    assert abs(qchem_rho_A_Nuc_B - embdic['nucArhoB']) < 1e-5
    assert abs(qchem_rho_B_Nuc_A - embdic['nucBrhoA']) < 1e-5
    # DFT related terms
    qchem_int_ref_xc = -0.0020002715
    qchem_int_ref_t = 0.0031792373
    qchem_exc_nad = -0.0031508737
    qchem_et_nad = 0.0038973367
    qchem_int_emb_xc = -0.0021589558
    qchem_int_emb_t = 0.0034058294
    qchem_deltalin = 0.0000679077
    assert abs(qchem_et_nad - embdic['et_nad']) < 1e-6
    assert abs(qchem_exc_nad - embdic['exc_nad']) < 1e-6
    assert abs(qchem_int_ref_t - embdic['int_ref_t']) < 1e-6
    assert abs(qchem_int_ref_xc - embdic['int_ref_xc']) < 1e-6
    assert abs(qchem_int_emb_t - embdic['int_emb_t']) < 1e-6
    assert abs(qchem_int_emb_xc - embdic['int_emb_xc']) < 1e-6
    assert abs(qchem_deltalin - embdic['deltalin']) < 1e-7


if __name__ == "__main__":
    run_co_h2o_psi4_sto3g()
    run_co_h2o_psi4_dz()
    run_co_h2o_psi4_tz()
    run_co_h2o_psi4_qz()
