import numpy
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
# dm_h2o = dm_h2o + dm_h2o.T
# CO
scfres2 = scf.RHF(co)
scfres2.kernel()
dm_co = scfres2.make_rdm1()
# dm_co = dm_co + dm_co.T


# Construct grid for complex
grids = gen_grid.Grids(system)
grids.level = 4
grids.build()
ao_h2o = eval_ao(h2o, grids.coords, deriv=0)
ao_co = eval_ao(co, grids.coords, deriv=0)

# Make Complex DM
ao_both = eval_ao(system, grids.coords, deriv=0)
nao_tot = h2o.nao_nr() + co.nao_nr()
dm_both = numpy.zeros((nao_tot, nao_tot))

nao_h2o = h2o.nao_nr()
dm_both[:nao_h2o, :nao_h2o] = dm_co
dm_both[nao_h2o:, nao_h2o:] = dm_h2o


rho_h2o  = eval_rho(h2o, ao_h2o, dm_h2o, xctype='LDA')
rho_co  = eval_rho(co, ao_co, dm_co, xctype='LDA')
#rho_both, dx_rho_both, dy_rho_both, dz_rho_both  = eval_rho(system, ao_both, dm_both, xctype='LDA')
#rho_diff = rho_both - rho_h2o
#vxcT_emb = libxc.eval_xc(xc_code, rho_both) 
xc_code = 'LDA'
vxcT_emb = libxc.eval_xc(xc_code, rho_h2o)
#vxcT_emb += libxc.eval_xc(t_code, rho_both) 
#vxcT_emb -= libxc.eval_xc(t_code, rho_h2o)
fock_emb = eval_mat(co, ao_co, grids.weights, rho_co, vxcT_emb, xctype='LDA')
#rho_co, dx_rho_co, dy_rho_co, dz_rho_co  = eval_rho(co, ao_co, dm_co, xctype='LDA')

#scfres2 = scf.RHF(co)
#scfres2.kernel()
