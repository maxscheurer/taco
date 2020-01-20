"""
Test method objects.
"""
import pytest
import numpy as np
from qcelemental.models import Molecule

from taco.methods.scf import ScfMethod
from taco.methods.scf_pyscf import ScfPyScf
from taco.testdata.cache import cache


def test_scfmethod():
    """Test base SCFMethod class."""
    mol = 'molecule'
    with pytest.raises(TypeError):
        ScfMethod(mol)
    mol = Molecule.from_data("""Li 0 0 0""")
    scf = ScfMethod(mol)
    with pytest.raises(NotImplementedError):
        scf.get_fock()
        scf.perturb_fock(np.zeros((2, 2)))
        scf.solve_scf()
    assert scf.density == None
    assert scf.energy == {}


def test_pyscf_base():
    """Test ScfPySCF class."""
    mol = Molecule.from_data("""Li 0 0 0""")
    mol2 = Molecule.from_data("""He 0 0 0""")
    basis = 0
    basis2 = 'sto-3g'
    method0 = 'adc'
    method2 = 'hf'
    method3 = 'dft'
    with pytest.raises(TypeError):
        hf = ScfPyScf(mol, basis, method2)
    with pytest.raises(ValueError):
        hf = ScfPyScf(mol2, basis2, method0)
        hf.perturb_fock(basis)
    with pytest.raises(TypeError):
        hf = ScfPyScf(mol2, basis2, method2)
        hf.perturb_fock(basis)
    with pytest.raises(ValueError):
        ScfPyScf(mol2, basis2, method3)
    with pytest.raises(NotImplementedError):
        ScfPyScf(mol, basis2, method2)


def test_hf_co_sto3g():
    """Test functions of ScfPyScf class."""
    mol = Molecule.from_data("""C        -3.6180905689    1.3768035675   -0.0207958979
                                O        -4.7356838533    1.5255563000    0.1150239130""")
    basis = 'sto-3g'
    method = 'hf'
    hf = ScfPyScf(mol, basis, method)
    hf.solve_scf(conv_tol=1e-12)
    dm0 = hf.get_density()
    nao_co = len(dm0)
    ref_dm0 = np.loadtxt(cache.files["co_h2o_sto3g_dma"]).reshape((nao_co, nao_co))
    np.testing.assert_allclose(ref_dm0*2, dm0, atol=1e-6)
    unperturbed_fock = hf.get_fock()
    assert 'scf' in hf.energy
    assert abs(hf.energy["scf"] - -111.22516947) < 1e-7
    vemb = np.zeros_like(dm0)
    hf.perturb_fock(vemb)
    hf.solve_scf()
    dm0_again = hf.get_density()
    np.testing.assert_allclose(ref_dm0*2, dm0_again, atol=1e-6)
    assert abs(hf.energy["scf"] - -111.22516947) < 1e-7
    perturbed_fock = hf.get_fock()
    np.testing.assert_allclose(unperturbed_fock, perturbed_fock, atol=1e-9)


def test_dft_co_sto3g():
    """Test functions of HFPySCF class."""
    mol = Molecule.from_data("""C        -3.6180905689    1.3768035675   -0.0207958979
                                O        -4.7356838533    1.5255563000    0.1150239130""")
    basis = 'sto-3g'
    method = 'dft'
    xc_code = 'LDA,VWN'
    dft = ScfPyScf(mol, basis, method, xc_code)
    dft.solve_scf(conv_tol=1e-12)
    dm0 = dft.get_density()
    unperturbed_fock = dft.get_fock()
    assert 'scf' in dft.energy
    assert abs(dft.energy["scf"] - -110.86517923) < 1e-5
    vemb = np.zeros_like(dm0)
    dft.perturb_fock(vemb)
    dft.solve_scf()
    assert abs(dft.energy["scf"] - -110.86517923) < 1e-5
    perturbed_fock = dft.get_fock()
    np.testing.assert_allclose(unperturbed_fock, perturbed_fock, atol=1e-9)


if __name__ == "__main__":
    test_scfmethod()
    test_pyscf_base()
    test_hf_co_sto3g()
    test_dft_co_sto3g()
