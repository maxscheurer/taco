"""Compute the embedding potential and export it."""
from qcelemental.models import Molecule

import adcc
from taco.embedding.pyscf_wrap import PyScfWrap

# Define Molecules with QCElemental
co = Molecule.from_data("""C        -3.6180905689    1.3768035675   -0.0207958979
                           O        -4.7356838533    1.5255563000    0.1150239130""")
h2o = Molecule.from_data("""O  -7.9563726699    1.4854060709    0.1167920007
                            H  -6.9923165534    1.4211335985    0.1774706091
                            H  -8.1058463545    2.4422204631    0.1115993752""")
# Define arguments
with open("/home/users/g/gonzalcr/projects/tests/fdeta/2water/cc-pvdz-segopt.nwchem", 'r') as fb:
    basis = fb.read()
basis = 'sto-3g'
method = 'hf'
xc_code = 'LDA,VWN'
args0 = {"mol": co, "basis": basis, "method": method}
args1 = {"mol": h2o, "basis": basis, "method": method}
embs = {"mol": co, "basis": basis, "method": 'hf',
        "xc_code": xc_code, "t_code": 'LDA_K_TF,'}
# Make a wrap
wrap = PyScfWrap(args0, args1, embs)
wrap.emb_method.scf_object.conv_tol = 1e-12
wrap.run_embedding()
# Get SCF object
scfres = wrap.emb_method.scf_object

# Call adcc
state = adcc.adc2(scfres, n_singlets=3)
print(state.describe())
