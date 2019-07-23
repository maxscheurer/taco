
import numpy as np

from pyscf import gto, scf
from pyscf.dft.numint import eval_ao, eval_rho, eval_mat
from pyscf.dft import gen_grid, libxc

# Run SCF in pyscf
h2o = gto.M(
    atom="""
            O  -7.9563726699    1.4854060709    0.1167920007
            H  -6.9923165534    1.4211335985    0.1774706091
            H  -8.1058463545    2.4422204631    0.1115993752
         """,
    basis='sto-3g',
)
co = gto.M(
    atom="""
            C  -3.6180905689    1.3768035675   -0.0207958979
            O  -4.7356838533    1.5255563000    0.1150239130
         """,
    basis='sto-3g',
        )
system = gto.M(atom=co.atom + h2o.atom)
# Get initial densities from HF
# H2O
# TODO: make a wrapper and make sure DMs are correct
scfres1 = scf.RHF(h2o)
scfres1.conv_tol = 1e-12
scfres1.kernel()
dm_h2o = scfres1.make_rdm1()

# CO
scfres2 = scf.RHF(co)
scfres2.conv_tol = 1e-12
scfres2.kernel()
dm_co = scfres2.make_rdm1()

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

dm_both[:nao_co, :nao_co] = dm_co
dm_both[nao_co:, nao_co:] = dm_h2o

# Compute all densities on a grid
rho_h2o = eval_rho(h2o, ao_h2o, dm_h2o, xctype='LDA')
rho_co = eval_rho(co, ao_co, dm_co, xctype='LDA')
rho_both = eval_rho(system, ao_both, dm_both, xctype='LDA')
xc_code = 'LDA,VWN'  # same as xc_code = 'XC_LDA_X + XC_LDA_C_VWN'
t_code = 'XC_LDA_K_TF'
ex, vxc, fxc, kxc = libxc.eval_xc(xc_code, rho_both)
ex2, vxc2, fxc2, kxc2 = libxc.eval_xc(xc_code, rho_co)
ex3, vxc3, fxc3, kxc3 = libxc.eval_xc(xc_code, rho_h2o)
eT, vT, fT, kT = libxc.eval_xc(t_code, rho_both)
eT2, vT2, fT2, kT2 = libxc.eval_xc(t_code, rho_co)
eT3, vT3, fT3, kT3 = libxc.eval_xc(t_code, rho_h2o)
vxc_emb = vxc[0] - vxc2[0]
vT_emb = vT[0] - vT2[0]
# Let's try something:
exc_final = np.dot(rho_both*grids.weights, ex)
exc_final -= np.dot(rho_co*grids.weights, ex2)
exc_final -= np.dot(rho_h2o*grids.weights, ex3)
eT_final = np.dot(rho_both*grids.weights, eT)
eT_final -= np.dot(rho_co*grids.weights, eT2)
eT_final -= np.dot(rho_h2o*grids.weights, eT3)

fock_emb_xc = eval_mat(co, ao_co, grids.weights, rho_co, vxc_emb, xctype='LDA')
fock_emb_T = eval_mat(co, ao_co, grids.weights, rho_co, vT_emb, xctype='LDA')

# Electrostatic part
# Coulomb repulsion
mol1234 = h2o + h2o + co + co
shls_slice = (0, h2o.nbas,
              h2o.nbas, h2o.nbas+h2o.nbas,
              h2o.nbas+h2o.nbas, h2o.nbas+h2o.nbas+co.nbas,
              h2o.nbas+h2o.nbas+co.nbas, mol1234.nbas)
eris = mol1234.intor('int2e', shls_slice=shls_slice)
v_coulomb = np.einsum('ab,abcd->cd', dm_h2o, eris)


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


# Nuclear-electron integrals
co_charges, co_coords = get_charges_and_coords(co)
h2o_charges, h2o_coords = get_charges_and_coords(h2o)
vAnucB = 0
for i, q in enumerate(h2o_charges):
    co.set_rinv_origin(h2o_coords[i])
    vAnucB += co.intor('int1e_rinv') * -q

vBnucA = 0
for i, q in enumerate(co_charges):
    h2o.set_rinv_origin(co_coords[i])
    vBnucA += h2o.intor('int1e_rinv') * -q

# Read reference
ref_dma = np.loadtxt("../data/co_h2o_sto3g_dma.txt").reshape((nao_co, nao_co))
ref_dmb = np.loadtxt("../data/co_h2o_sto3g_dmb.txt").reshape((nao_h2o, nao_h2o))
ref_fock_xc = np.loadtxt("../data/co_h2o_sto3g_vxc.txt").reshape((nao_co, nao_co))
ref_fock_T = np.loadtxt("../data/co_h2o_sto3g_vTs.txt").reshape((nao_co, nao_co))
ref_fock_vJ = np.loadtxt("../data/co_h2o_sto3g_vJ.txt").reshape((nao_co, nao_co))
ref_fock_vNuc0 = np.loadtxt("../data/co_h2o_sto3g_vNuc0.txt").reshape((nao_co, nao_co))
ref_fock_vNuc1 = np.loadtxt("../data/co_h2o_sto3g_vNuc1.txt").reshape((nao_h2o, nao_h2o))
np.testing.assert_allclose(ref_dma*2, dm_co, atol=1e-7)
np.testing.assert_allclose(ref_dmb*2, dm_h2o, atol=1e-7)
np.testing.assert_allclose(ref_fock_xc, fock_emb_xc, atol=1e-7)
np.testing.assert_allclose(ref_fock_T, fock_emb_T, atol=1e-7)
np.testing.assert_allclose(ref_fock_vJ, v_coulomb, atol=1e-7)
np.testing.assert_allclose(ref_fock_vNuc0, vAnucB, atol=1e-7)
np.testing.assert_allclose(ref_fock_vNuc1, vBnucA, atol=1e-7)

# Perform the HF-in-HF embedding
# Modify Fock matrix
focka_ref = scfres2.get_hcore()
focka = focka_ref.copy()
focka += fock_emb_T + fock_emb_xc + v_coulomb + vAnucB
scfres3 = scf.RHF(co)
scfres3.conv_tol = 1e-12
scfres3.get_hcore = lambda *args: focka

# Re-evaluate the energy
print("Compute again energy with non-additive part of the embedding potential")
scfres3.kernel()
# Get density matrix, to only evaluate
final_dma = scfres3.make_rdm1()

# Test for final energies
ref_scf_dma = np.loadtxt("../data/co_h2o_sto3g_final_dma.txt").reshape((nao_co, nao_co))
np.testing.assert_allclose(ref_scf_dma*2, final_dma, atol=1e-7)
qchem_rho_A_rho_B = 20.9457553682
qchem_rho_A_Nuc_B = -21.1298173325
qchem_rho_B_Nuc_A = -20.8957755874
qchem_Nuc_A_Nuc_B = 21.0776656185
qchem_int_ref_xc = -0.0011361532
qchem_int_ref_T = 0.0022364179

int_ref_xc = np.einsum('ab,ba', fock_emb_xc, dm_co)
int_ref_T = np.einsum('ab,ba', fock_emb_T, dm_co)
rhoArhoB = np.einsum('ab,ba', v_coulomb, final_dma)
nucArhoB = np.einsum('ab,ba', vAnucB, final_dma)
nucBrhoA = np.einsum('ab,ba', vBnucA, dm_h2o)
assert abs(qchem_rho_A_rho_B - rhoArhoB) < 1e-7
assert abs(qchem_rho_A_Nuc_B - nucArhoB) < 1e-7
assert abs(qchem_rho_B_Nuc_A - nucBrhoA) < 1e-7

# Linearization terms
qchem_int_emb_xc = -0.0011379466
qchem_int_emb_T = 0.0022398242
int_emb_xc = np.einsum('ab,ba', fock_emb_xc, final_dma)
int_emb_T = np.einsum('ab,ba', fock_emb_T, final_dma)
deltalin = (int_emb_xc - int_ref_xc) + (int_emb_T - int_ref_T)
assert abs(qchem_int_emb_T - int_emb_T) < 1e-7
assert abs(qchem_int_emb_xc - int_emb_xc) < 1e-7
assert abs(deltalin - 0.0000016129) < 1e-9
