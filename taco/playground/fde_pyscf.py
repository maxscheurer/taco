from pyscf import gto, scf

# Run SCF in pyscf
mol = gto.M(
    atom='O 0 0 0;'
         'H 0 0 1.795239827225189;'
         'H 1.693194615993441 0 -0.599043184453037',
    basis='sto-3g',
    unit="Bohr"
)
scfres = scf.RHF(mol)
scfres.conv_tol = 1e-10
scfres.conv_tol_grad = 1e-6
scfres.kernel()
