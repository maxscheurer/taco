import numpy as np

import psi4
# pySCF stuff
from pyscf import gto, scf
from pyscf.dft.numint import eval_ao, eval_rho, eval_mat
from pyscf.dft import gen_grid, libxc


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

psi4.set_options({'basis': 'sto-3g'})
e_a, wfn_a = psi4.energy("SCF", molecule=co, return_wfn=True)
bas_a = wfn_a.basisset()
psi4.core.clean()

psi4.set_options({'basis': 'sto-3g'})
e_b, wfn_b = psi4.energy("SCF", molecule=h2o, return_wfn=True)
dm_tot_b = wfn_b.Da().np + wfn_b.Db().np
bas_b = wfn_b.basisset()

mints = psi4.core.MintsHelper(wfn_a)
eri = mints.ao_eri(bas_b, bas_b, bas_a, bas_a)
v_j = np.einsum('ab,abcd->cd', dm_tot_b, eri)
print("Coulomb interaction potential")
print(v_j)

nuc_potential_b = psi4.core.ExternalPotential()
for i in range(h2o.natom()):
    geom = np.array([h2o.x(i), h2o.y(i), h2o.z(i)])
    if h2o.units() == 'Angstrom':
        geom *= psi4.constants.bohr2angstroms
    nuc_potential_b.addCharge(h2o.Z(i), *geom)
v_b = nuc_potential_b.computePotentialMatrix(bas_a)
print("Nuclear attraction")
print(v_b.np)

psi4system = build_supersystem(co, h2o)

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
eT, vT, fT, kT = libxc.eval_xc(t_code, rho_both)
eT2, vT2, fT2, kT2 = libxc.eval_xc(t_code, rho_co)
vxc_emb = vxc[0] - vxc2[0]
vT_emb = vT[0] - vT2[0]


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
