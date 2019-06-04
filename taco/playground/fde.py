import psi4

mol = psi4.geometry("""
0 1
O 0 0 0
H 0 0 1.795239827225189
H 1.693194615993441 0 -0.599043184453037
symmetry c1
units au
no_reorient
no_com
""")

psi4.core.be_quiet()
psi4.set_options({'basis': 'sto-3g',
                  'e_convergence': 1e-8,
                  'd_convergence': 1e-8})

e_a, wfn_a = psi4.energy("SCF", molecule=mol, return_wfn=True)
