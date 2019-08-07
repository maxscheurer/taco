"""
Test method objects.
"""
import os
import pytest
import pandas
import numpy as np
from qcelemental.models import Molecule

from taco.embedding.wrap import QCWrap
from taco.embedding.pyscf_wrap import PySCFWrap
from taco.testdata.cache import cache


def test_qcwrap():
    """Test base QCWrap class."""
    args0 = 'mol'
    args1 = 'mol'
    emb_args = 'mol'
    dict0 = {'mol': 0}
    with pytest.raises(TypeError):
        QCWrap(args0, args1, emb_args)
    with pytest.raises(TypeError):
        QCWrap(dict0, args1, emb_args)
    with pytest.raises(TypeError):
        QCWrap(dict0, dict0, emb_args)
    wrap = QCWrap(dict0, dict0, dict0)
    with pytest.raises(NotImplementedError):
        wrap.create_fragments(dict0, dict0)
    with pytest.raises(NotImplementedError):
        wrap.compute_embedding_potential()
    with pytest.raises(NotImplementedError):
        wrap.run_embedding()
    # Test the printing and export files functions
    # Print into file
    cwd = os.getcwd()
    wrap.energy_dict["nanana"] = 100.00
    wrap.print_embedding_information(to_csv=True)
    fname = os.path.join(cwd, 'embedding_energies.csv')
    fread = pandas.read_csv(fname)
    assert fread.columns == list(wrap.energy_dict)
    os.remove(fname)
    # Export file
    nao_co = 10
    ref_dm0 = np.loadtxt(cache.files["co_h2o_sto3g_dma"]).reshape((nao_co, nao_co))
    wrap.vemb_dict["dm0"] = ref_dm0
    cwd = os.getcwd()
    wrap.export_matrices()
    fname = os.path.join(cwd, 'dm0.txt')
    dm0 = np.loadtxt(fname)
    np.testing.assert_allclose(ref_dm0, dm0, atol=1e-10)
    os.remove(fname)


def test_pyscf_wrap0():
    """Test basic functionality of PySCFWrap."""
    mol = Molecule.from_data("""He 0 0 0""")
    basis = 'sto-3g'
    dict0 = {'mol': 0}
    args0 = {"mol": mol, "basis": basis, "method": 'adc'}
    args1 = {"mol": mol, "basis": basis, "method": 'dft'}
    embs0 = {"mol": mol, "basis": basis, "method": 'hf'}
    embs1 = {"mol": mol, "basis": basis, "method": 'hf',
             "xc_code": 'LDA,VWN', "t_code": 'XC_LDA_K_TF'}
    with pytest.raises(KeyError):
        PySCFWrap(dict0, embs0, embs1)
    with pytest.raises(KeyError):
        PySCFWrap(embs0, dict0, embs1)
    with pytest.raises(KeyError):
        PySCFWrap(embs0, args1, embs1)
    with pytest.raises(KeyError):
        PySCFWrap(embs0, embs0, embs0)
    with pytest.raises(NotImplementedError):
        PySCFWrap(args0, embs0, embs1)


def test_pyscf_wrap_hf_co_h2o_sto3g():
    """Test embedded HF-in-HF case."""
    # Compared with QChem results
    co = Molecule.from_data("""C        -3.6180905689    1.3768035675   -0.0207958979
                               O        -4.7356838533    1.5255563000    0.1150239130""")
    h2o = Molecule.from_data("""O  -7.9563726699    1.4854060709    0.1167920007
                                H  -6.9923165534    1.4211335985    0.1774706091
                                H  -8.1058463545    2.4422204631    0.1115993752""")
    basis = 'sto-3g'
    method = 'hf'
    args0 = {"mol": co, "basis": basis, "method": method}
    args1 = {"mol": h2o, "basis": basis, "method": method}
    embs = {"mol": co, "basis": basis, "method": 'hf',
            "xc_code": 'LDA,VWN', "t_code": 'XC_LDA_K_TF'}
    wrap = PySCFWrap(args0, args1, embs)
    vemb = wrap.compute_embedding_potential()
    nao_co = 10
    nao_h2o = 7
    matdic = wrap.vemb_dict
    # Read reference
    ref_fock_xc = np.loadtxt(cache.files["co_h2o_sto3g_vxc"]).reshape((nao_co, nao_co))
    ref_fock_t = np.loadtxt(cache.files["co_h2o_sto3g_vTs"]).reshape((nao_co, nao_co))
    ref_fock_vJ = np.loadtxt(cache.files["co_h2o_sto3g_vJ"]).reshape((nao_co, nao_co))
    ref_fock_vNuc0 = np.loadtxt(cache.files["co_h2o_sto3g_vNuc0"]).reshape((nao_co, nao_co))
    ref_fock_vNuc1 = np.loadtxt(cache.files["co_h2o_sto3g_vNuc1"]).reshape((nao_h2o, nao_h2o))
    np.testing.assert_allclose(ref_fock_xc, matdic['v_nad_xc'], atol=1e-7)
    np.testing.assert_allclose(ref_fock_t, matdic['v_nad_t'], atol=1e-7)
    np.testing.assert_allclose(ref_fock_vJ, matdic['v_coulomb'], atol=1e-7)
    np.testing.assert_allclose(ref_fock_vNuc0, matdic['v0_nuc1'], atol=1e-7)
    np.testing.assert_allclose(ref_fock_vNuc1, matdic['v1_nuc0'], atol=1e-7)
    np.testing.assert_allclose(ref_fock_vNuc0+ref_fock_t+ref_fock_xc+ref_fock_vJ, vemb, atol=1e-7)
    wrap.run_embedding()
    matdic = wrap.vemb_dict
    embdic = wrap.energy_dict
    ref_dma = np.loadtxt(cache.files["co_h2o_sto3g_dma"]).reshape((nao_co, nao_co))
    ref_dmb = np.loadtxt(cache.files["co_h2o_sto3g_dmb"]).reshape((nao_h2o, nao_h2o))
    ref_scf_dma = np.loadtxt(cache.files["co_h2o_sto3g_final_dma"]).reshape((nao_co, nao_co))
    np.testing.assert_allclose(ref_dma*2, matdic['dm0_ref'], atol=1e-6)
    np.testing.assert_allclose(ref_dmb*2, matdic['dm1_ref'], atol=1e-6)
    np.testing.assert_allclose(ref_scf_dma*2, matdic['dm0_final'], atol=2e-6)
    qchem_rho_A_rho_B = 20.9457553682
    qchem_rho_A_Nuc_B = -21.1298173325
    qchem_rho_B_Nuc_A = -20.8957755874
    assert abs(qchem_rho_A_rho_B - embdic['rho0_rho1']) < 1e-6
    assert abs(qchem_rho_A_Nuc_B - embdic['nuc0_rho1']) < 1e-6
    assert abs(qchem_rho_B_Nuc_A - embdic['nuc1_rho0']) < 1e-6
    # DFT related terms
    qchem_int_ref_xc = -0.0011361532
    qchem_int_ref_t = 0.0022364179
    qchem_exc_nad = -0.0021105605
    qchem_et_nad = 0.0030018734
    qchem_int_emb_xc = -0.0011379466
    qchem_int_emb_t = 0.0022398242
    qchem_deltalin = 0.0000016129
    assert abs(qchem_et_nad - embdic['et_nad']) < 1e-7
    assert abs(qchem_exc_nad - embdic['exc_nad']) < 1e-7
    assert abs(qchem_int_ref_t - embdic['int_ref_t']) < 1e-7
    assert abs(qchem_int_ref_xc - embdic['int_ref_xc']) < 1e-7
    assert abs(qchem_int_emb_t - embdic['int_emb_t']) < 1e-7
    assert abs(qchem_int_emb_xc - embdic['int_emb_xc']) < 1e-7
    assert abs(qchem_deltalin - embdic['deltalin']) < 1e-9


def test_pyscf_wrap_dft_co_h2o_sto3g():
    """Test embedded HF-in-HF case."""
    # Compared with QChem results
    co = Molecule.from_data("""C        -3.6180905689    1.3768035675   -0.0207958979
                               O        -4.7356838533    1.5255563000    0.1150239130""")
    h2o = Molecule.from_data("""O  -7.9563726699    1.4854060709    0.1167920007
                                H  -6.9923165534    1.4211335985    0.1774706091
                                H  -8.1058463545    2.4422204631    0.1115993752""")
    basis = 'sto-3g'
    xc_code = 'LDA,VWN'
    method = 'dft'
    args0 = {"mol": co, "basis": basis, "method": method, "xc_code": xc_code}
    args1 = {"mol": h2o, "basis": basis, "method": method, "xc_code": xc_code}
    embs = {"mol": co, "basis": basis, "method": 'dft',
            "xc_code": 'LDA,VWN', "t_code": 'XC_LDA_K_TF'}
    wrap = PySCFWrap(args0, args1, embs)
    wrap.run_embedding()
    embdic = wrap.energy_dict
    # Read reference
    qchem_rho_A_rho_B = 20.9016932248
    qchem_rho_A_Nuc_B = -21.0856319395
    qchem_rho_B_Nuc_A = -20.8950212739
    assert abs(qchem_rho_A_rho_B - embdic['rho0_rho1']) < 1e-5
    assert abs(qchem_rho_A_Nuc_B - embdic['nuc0_rho1']) < 1e-5
    assert abs(qchem_rho_B_Nuc_A - embdic['nuc1_rho0']) < 1e-5
    # DFT related terms
    qchem_int_ref_xc = -0.0011261095
    qchem_int_ref_t = 0.0022083882
    qchem_exc_nad = -0.0020907144
    qchem_et_nad = 0.0029633384
    qchem_int_emb_xc = -0.0011281762
    qchem_int_emb_t = 0.0022122190
    qchem_deltalin = 0.0000017641
    assert abs(qchem_et_nad - embdic['et_nad']) < 1e-6
    assert abs(qchem_exc_nad - embdic['exc_nad']) < 1e-6
    assert abs(qchem_int_ref_t - embdic['int_ref_t']) < 1e-6
    assert abs(qchem_int_ref_xc - embdic['int_ref_xc']) < 1e-6
    assert abs(qchem_int_emb_t - embdic['int_emb_t']) < 1e-6
    assert abs(qchem_int_emb_xc - embdic['int_emb_xc']) < 1e-6
    assert abs(qchem_deltalin - embdic['deltalin']) < 1e-7


if __name__ == "__main__":
    test_qcwrap()
    test_pyscf_wrap0()
    test_pyscf_wrap_hf_co_h2o_sto3g()
    test_pyscf_wrap_dft_co_h2o_sto3g()
