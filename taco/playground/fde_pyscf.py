
import numpy as np

from pyscf import gto, scf, dft
from pyscf import lib
from pyscf.dft.numint import eval_ao, eval_rho, eval_mat
from pyscf.dft import gen_grid, libxc

from taco.testdata.cache import cache


def get_charges_and_coords(mol):
    """Return arrays with charges and coordinates."""
    bohr2a = 0.52917721067
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
    # Coulomb repulsion
    mol1234 = mol2 + mol2 + mol1 + mol1
    shls_slice = (0, mol2.nbas,
                  mol2.nbas, mol2.nbas+mol2.nbas,
                  mol2.nbas+mol2.nbas, mol2.nbas+mol2.nbas+mol1.nbas,
                  mol2.nbas+mol2.nbas+mol1.nbas, mol1234.nbas)
    eris = mol1234.intor('int2e', shls_slice=shls_slice)
    v_coulomb = np.einsum('ab,abcd->cd', dm2, eris)
    return v_coulomb


def get_attraction_potential(mol1, mol2):
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
    # Evaluate energy densities and potentials on a grid
    exc, vxc, fxc, kxc = libxc.eval_xc(code, rho_both)
    exc2, vxc2, fxc2, kxc2 = libxc.eval_xc(code, rho1)
    exc3, vxc3, fxc3, kxc3 = libxc.eval_xc(code, rho2)
    return (exc, exc2, exc3), (vxc, vxc2, vxc3)


def get_nad_energy(grids, energies, rho_both, rho1, rho2):
    e_nad = np.dot(rho_both*grids.weights, energies[0])
    e_nad -= np.dot(rho1*grids.weights, energies[1])
    e_nad -= np.dot(rho2*grids.weights, energies[2])
    return e_nad


def run_co_h2o_pyscf(ibasis, return_matrices=False):
    # Run SCF in pyscf
    h2o = gto.M(
        atom="""
                O  -7.9563726699    1.4854060709    0.1167920007
                H  -6.9923165534    1.4211335985    0.1774706091
                H  -8.1058463545    2.4422204631    0.1115993752
             """,
        basis=ibasis,
    )
    co = gto.M(
        atom="""
                C  -3.6180905689    1.3768035675   -0.0207958979
                O  -4.7356838533    1.5255563000    0.1150239130
             """,
        basis=ibasis,
            )
    system = gto.M(atom=co.atom + h2o.atom, basis=ibasis)
    # Get initial densities from HF
    # H2O
    # TODO: make a wrapper and make sure DMs are correct
    scfres1 = scf.RHF(h2o)
    scfres1.conv_tol = 1e-12
    scfres1.kernel()
    dmb = scfres1.make_rdm1()

    # CO
    scfres2 = scf.RHF(co)
    scfres2.conv_tol = 1e-12
    scfres2.kernel()
    dma = scfres2.make_rdm1()

    # Construct grid for complex
    grids = gen_grid.Grids(system)
    grids.level = 4
    grids.build()
    ao_h2o = eval_ao(h2o, grids.coords, deriv=0)
    ao_co = eval_ao(co, grids.coords, deriv=0)

    # Make Complex DM
    ao_both = eval_ao(system, grids.coords, deriv=0)
    nao_co = co.nao_nr()
    nao_h2o = h2o.nao_nr()
    nao_tot = nao_co + nao_h2o
    dm_both = np.zeros((nao_tot, nao_tot))

    dm_both[:nao_co, :nao_co] = dma
    dm_both[nao_co:, nao_co:] = dmb

    # Compute DFT non-additive potential and energies
    rho_h2o = eval_rho(h2o, ao_h2o, dmb, xctype='LDA')
    rho_co = eval_rho(co, ao_co, dma, xctype='LDA')
    rho_both = eval_rho(system, ao_both, dm_both, xctype='LDA')
    # Compute all densities on a grid
    xc_code = 'LDA,VWN'  # same as xc_code = 'XC_LDA_X + XC_LDA_C_VWN'
    t_code = 'XC_LDA_K_TF'
    excs, vxcs = get_dft_grid_stuff(xc_code, rho_both, rho_co, rho_h2o)
    ets, vts = get_dft_grid_stuff(t_code, rho_both, rho_co, rho_h2o)
    vxc_emb = vxcs[0][0] - vxcs[1][0]
    vt_emb = vts[0][0] - vts[1][0]
    # Energy functionals:
    exc_nad = get_nad_energy(grids, excs, rho_both, rho_co, rho_h2o)
    et_nad = get_nad_energy(grids, ets, rho_both, rho_co, rho_h2o)

    fock_emb_xc = eval_mat(co, ao_co, grids.weights, rho_co, vxc_emb, xctype='LDA')
    fock_emb_t = eval_mat(co, ao_co, grids.weights, rho_co, vt_emb, xctype='LDA')

    # Electrostatic part
    v_coulomb = get_coulomb(co, h2o, dmb)

    # Nuclear-electron integrals
    vAnucB, vBnucA = get_attraction_potential(co, h2o)

    # Perform the HF-in-HF embedding
    # Modify Fock matrix
    focka_ref = scfres2.get_hcore()
    focka = focka_ref.copy()
    focka += fock_emb_t + fock_emb_xc + v_coulomb + vAnucB
    scfres3 = scf.RHF(co)
    scfres3.conv_tol = 1e-12
    scfres3.get_hcore = lambda *args: focka

    # Re-evaluate the energy
    scfres3.kernel()
    # Get density matrix, to only evaluate
    dma_final = scfres3.make_rdm1()

    int_ref_xc = np.einsum('ab,ba', fock_emb_xc, dma)
    int_ref_t = np.einsum('ab,ba', fock_emb_t, dma)
    rhoArhoB = np.einsum('ab,ba', v_coulomb, dma_final)
    nucArhoB = np.einsum('ab,ba', vAnucB, dma_final)
    nucBrhoA = np.einsum('ab,ba', vBnucA, dmb)

    # Linearization terms
    int_emb_xc = np.einsum('ab,ba', fock_emb_xc, dma_final)
    int_emb_t = np.einsum('ab,ba', fock_emb_t, dma_final)
    deltalin = (int_emb_xc - int_ref_xc) + (int_emb_t - int_ref_t)

    # Save terms in dictionary
    embdic = {}
    embdic['rhoArhoB'] = rhoArhoB
    embdic['nucArhoB'] = nucArhoB
    embdic['nucBrhoA'] = nucBrhoA
    embdic['exc_nad'] = exc_nad
    embdic['et_nad'] = et_nad
    embdic['int_ref_xc'] = int_ref_xc
    embdic['int_ref_t'] = int_ref_t
    embdic['int_emb_xc'] = int_emb_xc
    embdic['int_emb_t'] = int_emb_t
    embdic['deltalin'] = deltalin
    if return_matrices:
        matdic = {}
        matdic['dma'] = dma
        matdic['dmb'] = dmb
        matdic['dma_final'] = dma_final
        matdic['fock_emb_xc'] = fock_emb_xc
        matdic['fock_emb_t'] = fock_emb_t
        matdic['v_coulomb'] = v_coulomb
        matdic['vAnucB'] = vAnucB
        matdic['vBnucA'] = vBnucA
        return embdic, matdic
    else:
        return embdic


def run_co_h2o_pyscf_dft(ibasis, return_matrices=False):
    # Run SCF in pyscf
    h2o = gto.M(
        atom="""
                O  -7.9563726699    1.4854060709    0.1167920007
                H  -6.9923165534    1.4211335985    0.1774706091
                H  -8.1058463545    2.4422204631    0.1115993752
             """,
        basis=ibasis,
    )
    co = gto.M(
        atom="""
                C  -3.6180905689    1.3768035675   -0.0207958979
                O  -4.7356838533    1.5255563000    0.1150239130
             """,
        basis=ibasis,
            )
    system = gto.M(atom=co.atom + h2o.atom, basis=ibasis)
    # Get initial densities from HF
    # H2O
    # TODO: make a wrapper and make sure DMs are correct
    scfres1 = dft.RKS(h2o)
    scfres1.xc = 'LDA,VWN'
    scfres1.conv_tol = 1e-12
    scfres1.kernel()
    dmb = scfres1.make_rdm1()

    # CO
    scfres2 = dft.RKS(co)
    scfres2.xc = 'LDA,VWN'
    scfres2.conv_tol = 1e-12
    scfres2.kernel()
    dma = scfres2.make_rdm1()

    # Construct grid for complex
    grids = gen_grid.Grids(system)
    grids.level = 4
    grids.build()
    ao_h2o = eval_ao(h2o, grids.coords, deriv=0)
    ao_co = eval_ao(co, grids.coords, deriv=0)

    # Make Complex DM
    ao_both = eval_ao(system, grids.coords, deriv=0)
    nao_co = co.nao_nr()
    nao_h2o = h2o.nao_nr()
    nao_tot = nao_co + nao_h2o
    dm_both = np.zeros((nao_tot, nao_tot))

    dm_both[:nao_co, :nao_co] = dma
    dm_both[nao_co:, nao_co:] = dmb

    # Compute DFT non-additive potential and energies
    rho_h2o = eval_rho(h2o, ao_h2o, dmb, xctype='LDA')
    rho_co = eval_rho(co, ao_co, dma, xctype='LDA')
    rho_both = eval_rho(system, ao_both, dm_both, xctype='LDA')
    # Compute all densities on a grid
    xc_code = 'LDA,VWN'  # same as xc_code = 'XC_LDA_X + XC_LDA_C_VWN'
    t_code = 'XC_LDA_K_TF'
    excs, vxcs = get_dft_grid_stuff(xc_code, rho_both, rho_co, rho_h2o)
    ets, vts = get_dft_grid_stuff(t_code, rho_both, rho_co, rho_h2o)
    vxc_emb = vxcs[0][0] - vxcs[1][0]
    vt_emb = vts[0][0] - vts[1][0]
    # Energy functionals:
    exc_nad = get_nad_energy(grids, excs, rho_both, rho_co, rho_h2o)
    et_nad = get_nad_energy(grids, ets, rho_both, rho_co, rho_h2o)

    fock_emb_xc = eval_mat(co, ao_co, grids.weights, rho_co, vxc_emb, xctype='LDA')
    fock_emb_t = eval_mat(co, ao_co, grids.weights, rho_co, vt_emb, xctype='LDA')

    # Electrostatic part
    v_coulomb = get_coulomb(co, h2o, dmb)

    # Nuclear-electron integrals
    vAnucB, vBnucA = get_attraction_potential(co, h2o)

    # Perform the DFT-in-DFT embedding
    # Modify Fock matrix
    focka_ref = scfres2.get_hcore()
    focka = focka_ref.copy()
    focka += fock_emb_t + fock_emb_xc + v_coulomb + vAnucB
    scfres3 = dft.RKS(co)
    scfres3.xc = 'LDA,VWN'
    scfres3.conv_tol = 1e-12
    scfres3.get_hcore = lambda *args: focka

    # Re-evaluate the energy
    scfres3.kernel()
    # Get density matrix, to only evaluate
    dma_final = scfres3.make_rdm1()

    int_ref_xc = np.einsum('ab,ba', fock_emb_xc, dma)
    int_ref_t = np.einsum('ab,ba', fock_emb_t, dma)
    rhoArhoB = np.einsum('ab,ba', v_coulomb, dma_final)
    nucArhoB = np.einsum('ab,ba', vAnucB, dma_final)
    nucBrhoA = np.einsum('ab,ba', vBnucA, dmb)

    # Linearization terms
    int_emb_xc = np.einsum('ab,ba', fock_emb_xc, dma_final)
    int_emb_t = np.einsum('ab,ba', fock_emb_t, dma_final)
    deltalin = (int_emb_xc - int_ref_xc) + (int_emb_t - int_ref_t)

    # Save terms in dictionary
    embdic = {}
    embdic['rhoArhoB'] = rhoArhoB
    embdic['nucArhoB'] = nucArhoB
    embdic['nucBrhoA'] = nucBrhoA
    embdic['exc_nad'] = exc_nad
    embdic['et_nad'] = et_nad
    embdic['int_ref_xc'] = int_ref_xc
    embdic['int_ref_t'] = int_ref_t
    embdic['int_emb_xc'] = int_emb_xc
    embdic['int_emb_t'] = int_emb_t
    embdic['deltalin'] = deltalin
    if return_matrices:
        matdic = {}
        matdic['dma'] = dma
        matdic['dmb'] = dmb
        matdic['dma_final'] = dma_final
        matdic['fock_emb_xc'] = fock_emb_xc
        matdic['fock_emb_t'] = fock_emb_t
        matdic['v_coulomb'] = v_coulomb
        matdic['vAnucB'] = vAnucB
        matdic['vBnucA'] = vBnucA
        return embdic, matdic
    else:
        return embdic


def run_co_h2o_pyscf_sto3g():
    # Get HF-in-HF embedding information
    embdic, matdic = run_co_h2o_pyscf('sto-3g', True)
    nao_co = len(matdic['dma'])
    nao_h2o = len(matdic['dmb'])
    # Read reference
    ref_dma = np.loadtxt(cache.files["co_h2o_sto3g_dma"]).reshape((nao_co, nao_co))
    ref_dmb = np.loadtxt(cache.files["co_h2o_sto3g_dmb"]).reshape((nao_h2o, nao_h2o))
    ref_scf_dma = np.loadtxt(cache.files["co_h2o_sto3g_final_dma"]).reshape((nao_co, nao_co))
    ref_fock_xc = np.loadtxt(cache.files["co_h2o_sto3g_vxc"]).reshape((nao_co, nao_co))
    ref_fock_t = np.loadtxt(cache.files["co_h2o_sto3g_vTs"]).reshape((nao_co, nao_co))
    ref_fock_vJ = np.loadtxt(cache.files["co_h2o_sto3g_vJ"]).reshape((nao_co, nao_co))
    ref_fock_vNuc0 = np.loadtxt(cache.files["co_h2o_sto3g_vNuc0"]).reshape((nao_co, nao_co))
    ref_fock_vNuc1 = np.loadtxt(cache.files["co_h2o_sto3g_vNuc1"]).reshape((nao_h2o, nao_h2o))
    np.testing.assert_allclose(ref_dma*2, matdic['dma'], atol=1e-7)
    np.testing.assert_allclose(ref_dmb*2, matdic['dmb'], atol=1e-7)
    np.testing.assert_allclose(ref_scf_dma*2, matdic['dma_final'], atol=1e-7)
    np.testing.assert_allclose(ref_fock_xc, matdic['fock_emb_xc'], atol=1e-7)
    np.testing.assert_allclose(ref_fock_t, matdic['fock_emb_t'], atol=1e-7)
    np.testing.assert_allclose(ref_fock_vJ, matdic['v_coulomb'], atol=1e-7)
    np.testing.assert_allclose(ref_fock_vNuc0, matdic['vAnucB'], atol=1e-7)
    np.testing.assert_allclose(ref_fock_vNuc1, matdic['vBnucA'], atol=1e-7)
    qchem_rho_A_rho_B = 20.9457553682
    qchem_rho_A_Nuc_B = -21.1298173325
    qchem_rho_B_Nuc_A = -20.8957755874
    assert abs(qchem_rho_A_rho_B - embdic['rhoArhoB']) < 1e-7
    assert abs(qchem_rho_A_Nuc_B - embdic['nucArhoB']) < 1e-7
    assert abs(qchem_rho_B_Nuc_A - embdic['nucBrhoA']) < 1e-7
    # DFT related terms
    qchem_int_ref_xc = -0.0011361532
    qchem_int_ref_t = 0.0022364179
    qchem_exc_nad = -0.0021105605
    qchem_et_nad = 0.0030018734
    qchem_int_emb_xc = -0.0011379466
    qchem_int_emb_t = 0.0022398242
    qchem_deltalin = 0.0000016129
    assert abs(qchem_et_nad - embdic['et_nad']) < 1e-7
    assert abs(qchem_exc_nad - embdic['exc_nad']) < 1e-7
    assert abs(qchem_int_ref_t - embdic['int_ref_t']) < 1e-7
    assert abs(qchem_int_ref_xc - embdic['int_ref_xc']) < 1e-7
    assert abs(qchem_int_emb_t - embdic['int_emb_t']) < 1e-7
    assert abs(qchem_int_emb_xc - embdic['int_emb_xc']) < 1e-7
    assert abs(qchem_deltalin - embdic['deltalin']) < 1e-9


def run_co_h2o_pyscf_dz():
    # Get HF-in-HF embedding information
    embdic = run_co_h2o_pyscf('cc-pvdz')
    # Electronstatic terms
    # TODO: check why difference is not smaller
    qchem_rho_A_rho_B = 20.9054911884
    qchem_rho_A_Nuc_B = -21.1510526049
    qchem_rho_B_Nuc_A = -20.8349849585
    assert abs(qchem_rho_A_rho_B - embdic['rhoArhoB']) < 1e-6
    assert abs(qchem_rho_A_Nuc_B - embdic['nucArhoB']) < 1e-6
    assert abs(qchem_rho_B_Nuc_A - embdic['nucBrhoA']) < 1e-6
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
    assert abs(qchem_deltalin - embdic['deltalin']) < 1e-7


def run_co_h2o_pyscf_tz():
    # Get HF-in-HF embedding energies
    embdic = run_co_h2o_pyscf('cc-pvtz')
    # Electronstatic terms
    # TODO: check why difference is not smaller
    qchem_rho_A_rho_B = 20.9042017461
    qchem_rho_A_Nuc_B = -21.1526053452
    qchem_rho_B_Nuc_A = -20.8322820052
    assert abs(qchem_rho_A_rho_B - embdic['rhoArhoB']) < 1e-6
    assert abs(qchem_rho_A_Nuc_B - embdic['nucArhoB']) < 1e-6
    assert abs(qchem_rho_B_Nuc_A - embdic['nucBrhoA']) < 1e-6
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


def run_co_h2o_pyscf_qz():
    # Get HF-in-HF embedding energies
    embdic = run_co_h2o_pyscf('cc-pvqz')
    # Electronstatic terms
    # TODO: check why difference is not smaller
    qchem_rho_A_rho_B = 20.9070590943
    qchem_rho_A_Nuc_B = -21.1558918145
    qchem_rho_B_Nuc_A = -20.8320256245
    assert abs(qchem_rho_A_rho_B - embdic['rhoArhoB']) < 1e-6
    assert abs(qchem_rho_A_Nuc_B - embdic['nucArhoB']) < 1e-6
    assert abs(qchem_rho_B_Nuc_A - embdic['nucBrhoA']) < 1e-6
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


def run_co_h2o_pyscf_dft_sto3g():
    # Get DFT-in-DFT embedding energies
    embdic = run_co_h2o_pyscf_dft('sto-3g')
    # Electronstatic terms
    # TODO: check why difference is not smaller
    qchem_rho_A_rho_B = 20.9016932248
    qchem_rho_A_Nuc_B = -21.0856319395
    qchem_rho_B_Nuc_A = -20.8950212739
    assert abs(qchem_rho_A_rho_B - embdic['rhoArhoB']) < 1e-5
    assert abs(qchem_rho_A_Nuc_B - embdic['nucArhoB']) < 1e-5
    assert abs(qchem_rho_B_Nuc_A - embdic['nucBrhoA']) < 1e-5
    # DFT related terms
    qchem_int_ref_xc = -0.0011261095
    qchem_int_ref_t = 0.0022083882
    qchem_exc_nad = -0.0020907144
    qchem_et_nad = 0.0029633384
    qchem_int_emb_xc = -0.0011281762
    qchem_int_emb_t = 0.0022122190
    qchem_deltalin = 0.0000017641
    assert abs(qchem_et_nad - embdic['et_nad']) < 1e-6
    assert abs(qchem_exc_nad - embdic['exc_nad']) < 1e-6
    assert abs(qchem_int_ref_t - embdic['int_ref_t']) < 1e-6
    assert abs(qchem_int_ref_xc - embdic['int_ref_xc']) < 1e-6
    assert abs(qchem_int_emb_t - embdic['int_emb_t']) < 1e-6
    assert abs(qchem_int_emb_xc - embdic['int_emb_xc']) < 1e-6
    assert abs(qchem_deltalin - embdic['deltalin']) < 1e-7


def run_co_h2o_pyscf_dft_dz():
    # Get DFT-in-DFT embedding energies
    embdic = run_co_h2o_pyscf_dft('cc-pvdz')
    # Electronstatic terms
    # TODO: check why difference is not smaller
    qchem_rho_A_rho_B = 20.8725160406
    qchem_rho_A_Nuc_B = -21.1011016631
    qchem_rho_B_Nuc_A = -20.8507990282
    assert abs(qchem_rho_A_rho_B - embdic['rhoArhoB']) < 1e-5
    assert abs(qchem_rho_A_Nuc_B - embdic['nucArhoB']) < 1e-5
    assert abs(qchem_rho_B_Nuc_A - embdic['nucBrhoA']) < 1e-5
    # DFT related terms
    qchem_int_ref_xc = -0.0017325232
    qchem_int_ref_t = 0.0029539222
    qchem_exc_nad = -0.0028904861
    qchem_et_nad = 0.0037350854
    qchem_int_emb_xc = -0.0017855875
    qchem_int_emb_t = 0.0030332608
    qchem_deltalin = 0.0000262742
    assert abs(qchem_et_nad - embdic['et_nad']) < 1e-6
    assert abs(qchem_exc_nad - embdic['exc_nad']) < 1e-6
    assert abs(qchem_int_ref_t - embdic['int_ref_t']) < 1e-6
    assert abs(qchem_int_ref_xc - embdic['int_ref_xc']) < 1e-6
    assert abs(qchem_int_emb_t - embdic['int_emb_t']) < 1e-6
    assert abs(qchem_int_emb_xc - embdic['int_emb_xc']) < 1e-6
    assert abs(qchem_deltalin - embdic['deltalin']) < 1e-7


def run_co_h2o_pyscf_dft_tz():
    # Get DFT-in-DFT embedding energies
    embdic = run_co_h2o_pyscf_dft('cc-pvtz')
    # Electronstatic terms
    # TODO: check why difference is not smaller
    qchem_rho_A_rho_B = 20.8748968978
    qchem_rho_A_Nuc_B = -21.1114647888
    qchem_rho_B_Nuc_A = -20.8432761404
    assert abs(qchem_rho_A_rho_B - embdic['rhoArhoB']) < 1e-5
    assert abs(qchem_rho_A_Nuc_B - embdic['nucArhoB']) < 1e-5
    assert abs(qchem_rho_B_Nuc_A - embdic['nucBrhoA']) < 1e-5
    # DFT related terms
    qchem_int_ref_xc = -0.0021801275
    qchem_int_ref_t = 0.0035874689
    qchem_exc_nad = -0.0034828943
    qchem_et_nad = 0.0044317533
    qchem_int_emb_xc = -0.0022948480
    qchem_int_emb_t = 0.0037558987
    qchem_deltalin = 0.0000537094
    assert abs(qchem_et_nad - embdic['et_nad']) < 1e-6
    assert abs(qchem_exc_nad - embdic['exc_nad']) < 1e-6
    assert abs(qchem_int_ref_t - embdic['int_ref_t']) < 1e-6
    assert abs(qchem_int_ref_xc - embdic['int_ref_xc']) < 1e-6
    assert abs(qchem_int_emb_t - embdic['int_emb_t']) < 1e-6
    assert abs(qchem_int_emb_xc - embdic['int_emb_xc']) < 1e-6
    assert abs(qchem_deltalin - embdic['deltalin']) < 1e-7


def run_co_h2o_pyscf_dft_qz():
    # Get DFT-in-DFT embedding energies
    embdic = run_co_h2o_pyscf_dft('cc-pvqz')
    # Electronstatic terms
    # TODO: check why difference is not smaller
    qchem_rho_A_rho_B = 20.8783285224
    qchem_rho_A_Nuc_B = -21.1156265344
    qchem_rho_B_Nuc_A = -20.8427983360
    assert abs(qchem_rho_A_rho_B - embdic['rhoArhoB']) < 1e-5
    assert abs(qchem_rho_A_Nuc_B - embdic['nucArhoB']) < 1e-5
    assert abs(qchem_rho_B_Nuc_A - embdic['nucBrhoA']) < 1e-5
    # DFT related terms
    qchem_int_ref_xc = -0.0024337314
    qchem_int_ref_t = 0.0040024502
    qchem_exc_nad = -0.0038342924
    qchem_et_nad = 0.0049102990
    qchem_int_emb_xc = -0.0026336745
    qchem_int_emb_t = 0.0042939131
    qchem_deltalin = 0.0000915199
    assert abs(qchem_et_nad - embdic['et_nad']) < 1e-6
    assert abs(qchem_exc_nad - embdic['exc_nad']) < 1e-6
    assert abs(qchem_int_ref_t - embdic['int_ref_t']) < 1e-6
    assert abs(qchem_int_ref_xc - embdic['int_ref_xc']) < 1e-6
    assert abs(qchem_int_emb_t - embdic['int_emb_t']) < 1e-6
    assert abs(qchem_int_emb_xc - embdic['int_emb_xc']) < 1e-6
    assert abs(qchem_deltalin - embdic['deltalin']) < 1e-7


if __name__ == "__main__":
#   run_co_h2o_pyscf_sto3g()
#   run_co_h2o_pyscf_dz()
#   run_co_h2o_pyscf_tz()
#   run_co_h2o_pyscf_qz()
    run_co_h2o_pyscf_dft_sto3g()
    run_co_h2o_pyscf_dft_dz()
    run_co_h2o_pyscf_dft_tz()
    run_co_h2o_pyscf_dft_qz()
