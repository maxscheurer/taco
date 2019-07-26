"""
Test playground code for pyscf and psi4
"""
from taco.playground.fde_pyscf import run_co_h2o_pyscf
from taco.playground.fde_psi4 import run_co_h2o_psi4_sto3g, run_co_h2o_psi4_dz
from taco.playground.fde_psi4 import run_co_h2o_psi4_tz, run_co_h2o_psi4_qz


def test_playground_pyscf():
    run_co_h2o_pyscf()

def test_playground_psi4():
    run_co_h2o_psi4_sto3g()
    run_co_h2o_psi4_dz()
    run_co_h2o_psi4_tz()
    run_co_h2o_psi4_qz()
