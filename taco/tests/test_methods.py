"""
Test method objects.
"""
import numpy as np
from qcelemental.models import Molecule
import pytest

from taco.methods.scf import SCFMethod
from taco.methods.pyscf.hf import HFPySCF
from taco.methods.pyscf.dft import DFTPySCF
from taco.testdata.cache import cache


def test_scfmethod():
    """Test base SCFMethod class."""
    mol = 'molecule'
    with pytest.raises(TypeError):
        SCFMethod(mol)
    mol = Molecule.from_data("""Li 0 0 0""")
    scf = SCFMethod(mol)
    with pytest.raises(NotImplementedError):
        scf.get_fock()
        scf.perturbe_fock(np.zeros((2,2)))
        scf.solve()
    assert scf.density == None
    assert scf.energy == {}


def test_hfbase():
    """Test HFPySCF class."""
    mol = Molecule.from_data("""Li 0 0 0""")
    basis = 0
    with pytest.raises(TypeError):
        hf = HFPySCF(mol, basis)
    basis = 'sto-3g'
    with pytest.raises(NotImplementedError):
        hf = HFPySCF(mol, basis)

def test_hf_co_sto3g():
    """Test functions of HFPySCF class."""
    mol = Molecule.from_data("""C        -3.6180905689    1.3768035675   -0.0207958979
                                O        -4.7356838533    1.5255563000    0.1150239130""")
    basis = 'sto-3g'
    hf = HFPySCF(mol, basis)
    hf.solve()
    dm0 = hf.get_density()
    nao_co = len(dm0)
    ref_dm0 = np.loadtxt(cache.files["co_h2o_sto3g_dma"]).reshape((nao_co, nao_co))
    np.testing.assert_allclose(ref_dm0*2, dm0, atol=1e-6)
    assert 'scf' in hf.energy
    assert abs(hf.energy["scf"] - -111.22516947) < 1e-6
    vemb = np.zeros_like(dm0)
    hf.perturbe_fock(vemb)
    hf.solve()
    dm0_again = hf.get_density()
    np.testing.assert_allclose(ref_dm0*2, dm0_again, atol=1e-6)
    assert abs(hf.energy["scf"] - -111.22516947) < 1e-6

if __name__ == "__main__":
    test_scfmethod()
    test_hfbase()
    test_hf_co_sto3g()
