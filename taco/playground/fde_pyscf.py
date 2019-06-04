import numpy
from pyscf import gto, scf
from pyscf.dft.numint import eval_ao, eval_rho, eval_mat, _scale_ao, _dot_ao_ao, NumInt, nr_rks_vxc
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
system = gto.M(
    atom="""
            C  -3.6180905689    1.3768035675   -0.0207958979
            O  -4.7356838533    1.5255563000    0.1150239130
            O  -7.9563726699    1.4854060709    0.1167920007
            H  -6.9923165534    1.4211335985    0.1774706091
            H  -8.1058463545    2.4422204631    0.1115993752
         """,
    basis='sto-3g',
        )
# Get initial densities from HF
# H2O
# TODO: make a wrapper and make sure DMs are correct
scfres1 = scf.RHF(h2o)
scfres1.kernel()
dm_h2o = scfres1.make_rdm1()

# CO
scfres2 = scf.RHF(co)
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
dm_both = numpy.zeros((nao_tot, nao_tot))

dm_both[:nao_co, :nao_co] = dm_co
dm_both[nao_co:, nao_co:] = dm_h2o

# Compute all densities on a grid
rho_h2o  = eval_rho(h2o, ao_h2o, dm_h2o, xctype='LDA')
rho_co  = eval_rho(co, ao_co, dm_co, xctype='LDA')
rho_both  = eval_rho(system, ao_both, dm_both, xctype='LDA')
xc_code = 'LDA,VWN' # same as xc_code = 'XC_LDA_X + XC_LDA_C_VWN'
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
ref_dma = numpy.loadtxt("../data/co_h2o_sto3g_dma.txt").reshape((nao_co,nao_co))
ref_dmb = numpy.loadtxt("../data/co_h2o_sto3g_dmb.txt").reshape((nao_h2o,nao_h2o))
ref_fock_xc = numpy.loadtxt("../data/co_h2o_sto3g_vxc.txt").reshape((nao_co,nao_co))
ref_fock_T = numpy.loadtxt("../data/co_h2o_sto3g_vTs.txt").reshape((nao_co,nao_co))
assert numpy.allclose(ref_dma*2, dm_co, atol=1e-7)
assert numpy.allclose(ref_dmb*2, dm_h2o, atol=1e-7)
assert numpy.allclose(ref_fock_xc, fock_emb_xc)
assert numpy.allclose(ref_fock_T, fock_emb_T)
