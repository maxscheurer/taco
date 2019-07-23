import numpy as np

# pySCF stuff
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
dm_ref = dm_co.copy()

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


# Read reference
ref_dma = np.loadtxt("../data/co_h2o_sto3g_dma.txt").reshape((nao_co, nao_co))
ref_dmb = np.loadtxt("../data/co_h2o_sto3g_dmb.txt").reshape((nao_h2o, nao_h2o))
ref_fock_xc = np.loadtxt("../data/co_h2o_sto3g_vxc.txt").reshape((nao_co, nao_co))
ref_fock_T = np.loadtxt("../data/co_h2o_sto3g_vTs.txt").reshape((nao_co, nao_co))
ref_fock_vJ = np.loadtxt("../data/co_h2o_sto3g_vJ.txt").reshape((nao_co, nao_co))
ref_fock_vnuc0 = np.loadtxt("../data/co_h2o_sto3g_vNuc0.txt").reshape((nao_co, nao_co))
np.testing.assert_allclose(ref_dma*2, dm_co, atol=1e-7)
np.testing.assert_allclose(ref_dmb*2, dm_h2o, atol=1e-7)
np.testing.assert_allclose(ref_fock_xc, fock_emb_xc, atol=1e-7)
np.testing.assert_allclose(ref_fock_T, fock_emb_T, atol=1e-7)


# Modify the fock matrix of system 1
focka_ref = scfres2.get_hcore()
focka = focka_ref.copy()
int_fock = np.einsum('ab,ba', focka, dm_ref)
print("Integral of whole Fock before: ", int_fock)
non_elec_emb = ref_fock_xc + ref_fock_T
print("Non electrostatic part", non_elec_emb)
focka += ref_fock_T + ref_fock_xc + ref_fock_vJ + ref_fock_vnuc0
print("Fock after", focka)
scfres3 = scf.RHF(co)
scfres3.conv_tol = 1e-12
scfres3.get_hcore = lambda *args: focka
# Re-evaluate the energy
print("Compute again energy with non-additive part of the embedding potential")
scfres3.kernel()
# Get density matrix, to only evaluate
final_dma = scfres3.make_rdm1()
emb_energy = np.einsum('ab,ba', focka_ref, dm_ref)


# Correct the energy: subtract integral of the xcT potentials and add energy functionals
int_ref_xc = np.einsum('ab,ba', fock_emb_xc, dm_ref)
int_ref_T = np.einsum('ab,ba', fock_emb_T, dm_ref)
int_fock2 = np.einsum('ab,ba', focka, dm_ref)
print("Integral of xc potential : ", int_ref_xc)
print("Integral of Ts potential : ", int_ref_T)
print("Integral of whole Fock : ", int_fock2)
print("difference Fock integrals: ", int_fock2-int_fock)
print("Number of electrons: ", sum(rho_co*grids.weights))
print("E_xc[rho_ref] : ", exc_final)
print("E_T[rho_ref] : ", eT_final)
total_add = exc_final + eT_final

ENuc = 22.3674608413
E1 = -196.7776983134
EJ = 76.6269597500
Alpha_Exchange = -6.7209458758
Beta_Exchange = -6.7209458758
rho_A_rho_B = 20.9457553682
rho_A_Nuc_B = -21.1298173325
rho_B_Nuc_A = -20.8957755874
Nuc_A_Nuc_B = 21.0776656185

# Linearization terms
int_emb_xc = np.einsum('ab,ba', fock_emb_xc, final_dma)
int_emb_T = np.einsum('ab,ba', fock_emb_T, final_dma)
deltalin = (int_emb_xc - int_ref_xc) + (int_emb_T - int_ref_T)
print("DeltaLin term: ", deltalin)
print("What needs to be added to the energy:", total_add)
emb_energy += ENuc + EJ + Alpha_Exchange + Beta_Exchange
print("Final energy of embedded A system: ", emb_energy)
emb_energy += total_add + deltalin
print("Final, energy: ", emb_energy)
emb_energy += rho_A_rho_B + rho_A_Nuc_B + rho_B_Nuc_A + Nuc_A_Nuc_B
print("Final, very FINAL energy: ", emb_energy)
