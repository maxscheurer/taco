
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
    exc, vxc, fxc, kxc = libxc.eval_xc(code, rho_both, 0, 0, 2, 0)
    exc2, vxc2, fxc2, kxc2 = libxc.eval_xc(code, rho1, 0, 0, 2, 0)
    exc3, vxc3, fxc3, kxc3 = libxc.eval_xc(code, rho2, 0, 0, 2, 0)
    return (exc, exc2, exc3), (vxc, vxc2, vxc3), (fxc, fxc2, fxc3)


def get_nad_energy(grids, energies, rho_both, rho1, rho2):
    e_nad = np.dot(rho_both*grids.weights, energies[0])
    e_nad -= np.dot(rho1*grids.weights, energies[1])
    e_nad -= np.dot(rho2*grids.weights, energies[2])
    return e_nad


def run_co_h2o_pyscf_tddft(ibasis, return_matrices=False):
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
    grids = gen_grid.Grids(co)
    #grids = scfres2.grids
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
    excs, vxcs, fxcs = get_dft_grid_stuff(xc_code, rho_both, rho_co, rho_h2o)
    ets, vts, fts = get_dft_grid_stuff(t_code, rho_both, rho_co, rho_h2o)
    # Non-additive terms
    # Potential
    vxc_emb = vxcs[0][0] - vxcs[1][0]
    vt_emb = vts[0][0] - vts[1][0]
    # Kernel
    fxc_emb = fxcs[0][0] - fxcs[1][0]
    ft_emb = fts[0][0] - fts[1][0]
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

    # Perform TDDFT
    vxct_emb = (vxc_emb + vt_emb, None, None, None)
    fxct_emb = (fxc_emb + ft_emb, None, None, None, None, None, None, None, None, None)
    from pyscf.tddft import rks
    from pyscf.tdscf import TDDFT

    # Use same grid
    scfres2.grids = grids
    scfres3.grids = grids
    # Molecule alone
    print("Isolated CO")
    td0 = TDDFT(scfres2)
    es0 = td0.kernel(nstates=5)[0] * 27.2114
    # with modified potential
    print("CO with embedded vxct")
    td1 = TDDFT(scfres3)
    es1 = td1.kernel(nstates=5)[0] * 27.2114
    # Embedded
    print("CO with embedded vxct and fxct")
    td = rks.FDETDDFT(scfres3, vxct_emb, fxct_emb)
    es = td.kernel(nstates=5)[0] * 27.2114
    # Embedded but only fxc
    print("CO with embedded fxct")
    td2 = rks.FDETDDFT(scfres2, vxct_emb, fxct_emb)
    es2 = td2.kernel(nstates=5)[0] * 27.2114
    print("CO TDA with embedded vxct and fxct")
    td3 = rks.FDETDA(scfres3, vxct_emb, fxct_emb)
    es3 = td3.kernel(nstates=5)[0] * 27.2114

 #  int_ref_xc = np.einsum('ab,ba', fock_emb_xc, dma)
 #  int_ref_t = np.einsum('ab,ba', fock_emb_t, dma)
 #  rhoArhoB = np.einsum('ab,ba', v_coulomb, dma_final)
 #  nucArhoB = np.einsum('ab,ba', vAnucB, dma_final)
 #  nucBrhoA = np.einsum('ab,ba', vBnucA, dmb)

 #  # Linearization terms
 #  int_emb_xc = np.einsum('ab,ba', fock_emb_xc, dma_final)
 #  int_emb_t = np.einsum('ab,ba', fock_emb_t, dma_final)
 #  deltalin = (int_emb_xc - int_ref_xc) + (int_emb_t - int_ref_t)

 #  # Save terms in dictionary
 #  embdic = {}
 #  embdic['rhoArhoB'] = rhoArhoB
 #  embdic['nucArhoB'] = nucArhoB
 #  embdic['nucBrhoA'] = nucBrhoA
 #  embdic['exc_nad'] = exc_nad
 #  embdic['et_nad'] = et_nad
 #  embdic['int_ref_xc'] = int_ref_xc
 #  embdic['int_ref_t'] = int_ref_t
 #  embdic['int_emb_xc'] = int_emb_xc
 #  embdic['int_emb_t'] = int_emb_t
 #  embdic['deltalin'] = deltalin
 #  if return_matrices:
 #      matdic = {}
 #      matdic['dma'] = dma
 #      matdic['dmb'] = dmb
 #      matdic['dma_final'] = dma_final
 #      matdic['fock_emb_xc'] = fock_emb_xc
 #      matdic['fock_emb_t'] = fock_emb_t
 #      matdic['v_coulomb'] = v_coulomb
 #      matdic['vAnucB'] = vAnucB
 #      matdic['vBnucA'] = vBnucA
 #      return embdic, matdic
 #  else:
 #      return embdic


def run_co_h2o_pyscf_tddft_sto3g():
    # Get HF-in-HF embedding information
    print("========== STO-3G =============")
    run_co_h2o_pyscf_tddft('sto-3g', True)
    print("========== CC-PVDZ =============")
    run_co_h2o_pyscf_tddft('cc-pvdz', True)
    print("========== CC-PVTZ =============")
    run_co_h2o_pyscf_tddft('cc-pvtz', True)


if __name__ == "__main__":
    run_co_h2o_pyscf_tddft_sto3g()
