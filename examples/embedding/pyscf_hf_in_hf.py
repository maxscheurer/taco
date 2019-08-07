"""Run embedded HF-in-HF case."""
from qcelemental.models import Molecule

from taco.embedding.pyscf_wrap import PyScfWrap

# Define Molecules with QCElemental
co = Molecule.from_data("""C        -3.6180905689    1.3768035675   -0.0207958979
                           O        -4.7356838533    1.5255563000    0.1150239130""")
h2o = Molecule.from_data("""O  -7.9563726699    1.4854060709    0.1167920007
                            H  -6.9923165534    1.4211335985    0.1774706091
                            H  -8.1058463545    2.4422204631    0.1115993752""")
# Define arguments
basis = 'cc-pvdz'
method = 'hf'
args0 = {"mol": co, "basis": basis, "method": method}
args1 = {"mol": h2o, "basis": basis, "method": method}
# Use LDA functionals for embedding potential, and solve final energy with HF
embs = {"mol": co, "basis": basis, "method": 'hf',
        "xc_code": 'LDA,VWN', "t_code": 'XC_LDA_K_TF'}
# Make a wrap
wrap = PyScfWrap(args0, args1, embs)
# Run the embedding calculation
wrap.run_embedding()
# Save information to files
wrap.print_embedding_information(to_csv=True)
wrap.export_matrices()
