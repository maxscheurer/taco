"""
Test method objects.
"""
import os
import pytest
import pandas
import numpy as np
from qcelemental.models import Molecule
from pyscf import gto
from pyscf.dft import gen_grid

from taco.embedding.embpot import EmbPotBase
from taco.embedding.pyscf_embpot import PyScfEmbPot
from taco.embedding.scf_wrap import ScfWrap
from taco.embedding.scf_wrap_single import ScfWrapSingle
from taco.embedding.pyscf_tddft import compute_emb_kernel
from taco.embedding.pyscf_wrap import PyScfWrap
from taco.embedding.pyscf_wrap_single import PyScfWrapSingle
from taco.embedding.postscf_wrap import PostScfWrap
from taco.embedding.omolcas_wrap import OpenMolcasWrap
from taco.testdata.cache import cache


def test_embpotbase():
    """Test EmbPotBase class."""
    mol0 = 'mol'
    mol1 = 'mol'
    emb_args = 'mol'
    dict0 = {'mol': 0}
    dict1 = {'xc_code': 0}
    with pytest.raises(TypeError):
        EmbPotBase(mol0, mol1, emb_args)
    with pytest.raises(KeyError):
        EmbPotBase(mol0, mol1, dict0)
    with pytest.raises(KeyError):
        EmbPotBase(mol0, mol1, dict1)
    dict2 = {'xc_code': 0, 't_code': 0}
    # Check assign_dm
    pot = EmbPotBase(mol0, mol1, dict2)
    with pytest.raises(ValueError):
        pot.assign_dm(2, 0)
    with pytest.raises(TypeError):
        pot.assign_dm(0, 0)
    with pytest.raises(NotImplementedError):
        pot.save_maininfo(mol0)
    with pytest.raises(NotImplementedError):
        pot.compute_coulomb_potential()
    with pytest.raises(NotImplementedError):
        pot.compute_attraction_potential()
    with pytest.raises(NotImplementedError):
        pot.compute_nad_potential()
    with pytest.raises(NotImplementedError):
        pot.compute_embedding_potential(0, 1)


def test_pyscfembpot0():
    """Basic Tests for PyScfEmbPot class."""
    pyscfmol = gto.M(atom="""He  0.000   0.000   0.000""",
                     basis='sto-3g')
    mol0 = 'mol'
    mol1 = 'mol'
    args = 'mol'
    dict0 = {'mol': 0}
    dict1 = {'xc_code': 0}
    with pytest.raises(TypeError):
        PyScfEmbPot(mol0, mol1, args)
    with pytest.raises(TypeError):
        PyScfEmbPot(pyscfmol, mol1, args)
    with pytest.raises(KeyError):
        PyScfEmbPot(pyscfmol, pyscfmol, dict0)
    with pytest.raises(KeyError):
        PyScfEmbPot(pyscfmol, pyscfmol, dict1)
    emb_args = {'xc_code': 0, 't_code': 0}
    # Check assign_dm
    pot = PyScfEmbPot(pyscfmol, pyscfmol, emb_args)
    with pytest.raises(AttributeError):
        pot.compute_embedding_potential()
    dm = np.ones((4, 4))
    with pytest.raises(AttributeError):
        pot.compute_embedding_potential(dm0=dm)


def test_pyscf_embpot_hf_co_h2o_sto3g():
    """Test embedded HF-in-HF case."""
    # Compared with ScfWrap results
    basis = 'sto-3g'
    co = gto.M(atom="""C        -3.6180905689    1.3768035675   -0.0207958979
                       O        -4.7356838533    1.5255563000    0.1150239130""",
               basis=basis)
    h2o = gto.M(atom="""O  -7.9563726699    1.4854060709    0.1167920007
                        H  -6.9923165534    1.4211335985    0.1774706091
                        H  -8.1058463545    2.4422204631    0.1115993752""",
                basis=basis)
    nao_co = 10
    nao_h2o = 7
    ref_dma = 2*np.loadtxt(cache.files["co_h2o_sto3g_dma"]).reshape((nao_co, nao_co))
    ref_dmb = 2*np.loadtxt(cache.files["co_h2o_sto3g_dmb"]).reshape((nao_h2o, nao_h2o))
    embs = {"xc_code": 'LDA,VWN', "t_code": 'LDA_K_TF,'}
    embpot = PyScfEmbPot(co, h2o, embs)
    vemb = embpot.compute_embedding_potential(ref_dma, ref_dmb)
    matdic = embpot.vemb_dict
    # Read reference
    ref_xc = np.loadtxt(cache.files["co_h2o_sto3g_vxc"]).reshape((nao_co, nao_co))
    ref_t = np.loadtxt(cache.files["co_h2o_sto3g_vTs"]).reshape((nao_co, nao_co))
    ref_vJ = np.loadtxt(cache.files["co_h2o_sto3g_vJ"]).reshape((nao_co, nao_co))
    ref_vNuc0 = np.loadtxt(cache.files["co_h2o_sto3g_vNuc0"]).reshape((nao_co, nao_co))
    ref_vNuc1 = np.loadtxt(cache.files["co_h2o_sto3g_vNuc1"]).reshape((nao_h2o, nao_h2o))
    np.testing.assert_allclose(ref_xc, matdic['v_nad_xc'], atol=1e-7)
    np.testing.assert_allclose(ref_t, matdic['v_nad_t'], atol=1e-7)
    np.testing.assert_allclose(ref_vJ, matdic['v_coulomb'], atol=1e-7)
    np.testing.assert_allclose(ref_vNuc0, matdic['v0_nuc1'], atol=1e-7)
    np.testing.assert_allclose(ref_vNuc1, matdic['v1_nuc0'], atol=1e-7)
    np.testing.assert_allclose(ref_vNuc0+ref_t+ref_xc+ref_vJ, vemb, atol=1e-7)


def test_scfwrap():
    """Test base ScfWrap class."""
    args0 = 'mol'
    args1 = 'mol'
    emb_args = 'mol'
    dict0 = {'mol': 0}
    with pytest.raises(TypeError):
        ScfWrap(args0, args1, emb_args)
    with pytest.raises(TypeError):
        ScfWrap(dict0, args1, emb_args)
    with pytest.raises(TypeError):
        ScfWrap(dict0, dict0, emb_args)
    wrap = ScfWrap(dict0, dict0, dict0)
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
    """Test basic functionality of PyScfWrap."""
    mol = Molecule.from_data("""He 0 0 0""")
    basis = 'sto-3g'
    dict0 = {'mol': 0}
    args0 = {"mol": mol, "basis": basis, "method": 'adc'}
    args1 = {"mol": mol, "basis": basis, "method": 'dft'}
    embs0 = {"mol": mol, "basis": basis, "method": 'hf'}
    embs1 = {"mol": mol, "basis": basis, "method": 'hf',
             "xc_code": 'LDA,VWN', "t_code": 'LDA_K_TF,'}
    with pytest.raises(KeyError):
        PyScfWrap(dict0, embs0, embs1)
    with pytest.raises(KeyError):
        PyScfWrap(embs0, dict0, embs1)
    with pytest.raises(ValueError):
        PyScfWrap(embs0, args1, embs1)
    with pytest.raises(KeyError):
        PyScfWrap(embs0, embs0, embs0)
    with pytest.raises(ValueError):
        PyScfWrap(args0, embs0, embs1)


def test_pyscf_wrap_hf_co_h2o_sto3g():
    """Test embedded HF-in-HF case."""
    # Compared with ScfWrap results
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
            "xc_code": 'LDA,VWN', "t_code": 'LDA_K_TF,'}
    wrap = PyScfWrap(args0, args1, embs)
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
            "xc_code": 'LDA,VWN', "t_code": 'LDA_K_TF,'}
    wrap = PyScfWrap(args0, args1, embs)
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


def test_scfwrap_single():
    """Test base ScfWrap class."""
    args0 = 'mol'
    emb_args = 0.7
    dict0 = {'mol': 0}

    def fn0(r):
        """Little dummy function."""
        return np.power(r, 2)

    with pytest.raises(TypeError):
        ScfWrapSingle(args0, dict0, fn0, dict0, dict0)
    with pytest.raises(TypeError):
        ScfWrapSingle(dict0, args0, fn0, dict0, dict0)
    with pytest.raises(TypeError):
        ScfWrapSingle(dict0, dict0, args0, dict0, dict0)
    with pytest.raises(TypeError):
        ScfWrapSingle(dict0, dict0, fn0, emb_args, dict0)
    with pytest.raises(TypeError):
        ScfWrapSingle(dict0, dict0, fn0, dict0, args0)
    # Create the fake object and test base functions
    wrap = ScfWrapSingle(dict0, dict0, fn0, dict0, dict0)
    with pytest.raises(NotImplementedError):
        wrap.create_fragment(dict0)
    with pytest.raises(NotImplementedError):
        wrap.compute_embedding_potential()
    with pytest.raises(NotImplementedError):
        wrap.run_embedding()
    with pytest.raises(NotImplementedError):
        wrap.save_info()
    # Test checking arguments
    args0 = dict(basis='a', method='b', xc_code='c')
    with pytest.raises(KeyError):
        wrap.check_qc_arguments(args0)
    args0 = dict(mol='a', method='b', xc_code='c')
    with pytest.raises(KeyError):
        wrap.check_qc_arguments(args0)
    args0 = dict(mol=0.7, basis='a', xc_code='c')
    with pytest.raises(KeyError):
        wrap.check_qc_arguments(args0)
    args0 = dict(mol=0.7, basis='a', method='c')
    with pytest.raises(KeyError):
        wrap.check_emb_arguments(args0)
    args0 = dict(mol=0.7, basis='a', method='c', xc_code='d')
    with pytest.raises(KeyError):
        wrap.check_emb_arguments(args0)
    charge_args = dict(charges_coords=7)
    with pytest.raises(KeyError):
        wrap.check_emb_arguments(charge_args)
    charge_args = dict(charges=7)
    with pytest.raises(KeyError):
        wrap.check_emb_arguments(charge_args)


def test_pyscf_wrap_single_co_h2o():
    from taco.methods.scf_pyscf import get_pyscf_molecule
    # Create real object and test the cheking functions
    co = Molecule.from_data("""C        -3.6180905689    1.3768035675   -0.0207958979
                               O        -4.7356838533    1.5255563000    0.1150239130""")
    basis = 'sto-3g'
    xc_code = 'LDA,VWN'
    method = 'dft'
    args0 = {"mol": co, "basis": basis, "method": method, "xc_code": xc_code}
    embs = {"mol": co, "basis": basis, "method": 'dft',
            "xc_code": 'LDA,VWN', "t_code": 'LDA_K_TF,'}
    h2o_coords = np.array([[-7.9563726699, 1.4854060709, 0.1167920007],
                           [-6.9923165534, 1.4211335985, 0.1774706091],
                           [-8.1058463545, 2.4422204631, 0.1115993752]])
    h2o_charges = np.array([8., 1., 1.])
    frag_charges = dict(charges=h2o_charges, charges_coords=h2o_coords)

    def fn0(r):
        """Little dummy function."""

        return np.einsum('ab->a', r)

    # Make molecule in pyscf
    pyscfmol = get_pyscf_molecule(co, basis)
    # Construct grid for integration
    grids = gen_grid.Grids(pyscfmol)
    grids.level = 4
    grids.build()
    grid_args = dict(points=grids.coords, weights=grids.weights)
    wrap1 = PyScfWrapSingle(args0, frag_charges, fn0, grid_args, embs)
    emb_pot = wrap1.compute_embedding_potential()
    print("Embedding potential: \n", emb_pot)


def test_postscfwrap():
    """Test base PostScfWrap class."""
    pot0 = 'mol'
    dict0 = {'mol': 0}
    emb_args = {'xc_code': 0, 't_code': 0}
    emb_pot = EmbPotBase(dict0, dict0, emb_args)
    with pytest.raises(TypeError):
        PostScfWrap(pot0)
    wrap = PostScfWrap(emb_pot)
    with pytest.raises(NotImplementedError):
        wrap.format_potential()
    with pytest.raises(NotImplementedError):
        wrap.get_density()
    with pytest.raises(ValueError):
        wrap.save_info()
    # Test the printing and export files functions
    # Print into file
    cwd = os.getcwd()
    wrap.energy_dict["nanana"] = 100.00
    wrap.print_embedding_information(to_csv=True)
    fname = os.path.join(cwd, 'postscf_embedding_energies.csv')
    fread = pandas.read_csv(fname)
    assert fread.columns == list(wrap.energy_dict)
    os.remove(fname)
    # Export file
    nao_co = 10
    ref_dm0 = np.loadtxt(cache.files["co_h2o_sto3g_dma"]).reshape((nao_co, nao_co))
    wrap.dms_dict["dm0"] = ref_dm0
    cwd = os.getcwd()
    wrap.export_matrices()
    fname = os.path.join(cwd, 'dm0.txt')
    dm0 = np.loadtxt(fname)
    np.testing.assert_allclose(ref_dm0, dm0, atol=1e-10)
    os.remove(fname)


def test_postscfwrap_co_h2o():
    """Test energy and array part of the class."""
    # Compared with ScfWrap results
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
            "xc_code": 'LDA,VWN', "t_code": 'LDA_K_TF,'}
    wrap = PyScfWrap(args0, args1, embs)
    wrap.run_embedding()
    emb_pot = wrap.pot_object
    matdic = wrap.vemb_dict
    embdic = wrap.energy_dict
    postwrap = PostScfWrap(emb_pot)
    postwrap.dms_dict["dm0_final"] = matdic["dm0_final"]
    with pytest.raises(KeyError):
        postwrap.save_info()
    postwrap.emb_pot.vemb_dict["v_nad_xc_final"] = wrap.vemb_dict["v_nad_xc_final"]
    with pytest.raises(KeyError):
        postwrap.save_info()
    postwrap.emb_pot.vemb_dict["v_nad_t_final"] = wrap.vemb_dict["v_nad_t_final"]
    with pytest.raises(KeyError):
        postwrap.save_info()
    postwrap.emb_pot.vemb_dict["int_ref_xc"] = wrap.energy_dict["int_ref_xc"]
    with pytest.raises(KeyError):
        postwrap.save_info()
    postwrap.emb_pot.vemb_dict["int_ref_t"] = wrap.energy_dict["int_ref_t"]
    postwrap = PostScfWrap(emb_pot)
    postwrap.dms_dict["dm0_final"] = matdic["dm0_final"]
    postwrap.prepare_for_postscf(embdic, matdic)
    postwrap.save_info()
    assert abs(postwrap.energy_dict['et_nad_final'] - embdic['et_nad_final']) < 1e-6
    assert abs(postwrap.energy_dict['exc_nad_final'] - embdic['exc_nad_final']) < 1e-6
    assert abs(postwrap.energy_dict['int_final_xc'] - embdic['int_emb_xc']) < 1e-6
    assert abs(postwrap.energy_dict['int_final_t'] - embdic['int_emb_t']) < 1e-6
    assert abs(postwrap.energy_dict['int_emb_xc'] - embdic['int_emb_xc']) < 1e-6
    assert abs(postwrap.energy_dict['int_emb_t'] - embdic['int_emb_t']) < 1e-6
    assert abs(postwrap.energy_dict['deltalin'] - embdic['deltalin']) < 1e-6


def test_omolcas_wrap0():
    # Compared with ScfWrap results
    # TODO: rewrite basic test!
    return


def test_omolcas_wrap_co_h2o_ccpvdz():
    # Compared with ScfWrap results
    # Compared with ScfWrap results
    co = Molecule.from_data("""C        -3.6180905689    1.3768035675   -0.0207958979
                               O        -4.7356838533    1.5255563000    0.1150239130""")
    h2o = Molecule.from_data("""O  -7.9563726699    1.4854060709    0.1167920007
                                H  -6.9923165534    1.4211335985    0.1774706091
                                H  -8.1058463545    2.4422204631    0.1115993752""")
    with open(cache.files["molcas_basis_cc-pvdz"], 'r') as bfile:
        basis = bfile.read()
    method = 'hf'
    args0 = {"mol": co, "basis": basis, "method": method}
    args1 = {"mol": h2o, "basis": basis, "method": method}
    embs = {"mol": co, "basis": basis, "method": 'hf',
            "xc_code": 'LDA,VWN', "t_code": 'LDA_K_TF,'}
    wrap = PyScfWrap(args0, args1, embs)
    wrap.run_embedding()
    wrap.print_embedding_information()
    emb_pot = wrap.pot_object
    emb_pot.maininfo['basis'] = 'cc-pvdz'
    matdic = wrap.vemb_dict
    embdic = wrap.energy_dict
    postwrap = OpenMolcasWrap(emb_pot)
    postwrap.format_potential()
    vemb_tri = np.loadtxt('vemb_ordered_tri.txt')
    ref_pot = np.loadtxt(cache.files["molcas_vemb_co_h2o_cc-pvdz"])
    np.testing.assert_allclose(ref_pot, vemb_tri, atol=1e-4)
    postwrap.prepare_for_postscf(embdic, matdic)
    fin_dm = np.copy(matdic['dm0_final'])
    read_dm = postwrap.get_density(cache.files["molcas_runascii_co_h2o_cc-pvdz"])
    np.testing.assert_allclose(fin_dm, read_dm, atol=1e-5)
    postwrap.dms_dict['dm0_final'] = read_dm
    postwrap.save_info()
    assert abs(postwrap.energy_dict['et_nad_final'] - embdic['et_nad_final']) < 1e-6
    assert abs(postwrap.energy_dict['exc_nad_final'] - embdic['exc_nad_final']) < 1e-6
    total_emb_ref = embdic['int_emb_xc'] + embdic['int_emb_t']
    total_emb_new = postwrap.energy_dict['int_emb_xc'] + postwrap.energy_dict['int_emb_t']
    print(total_emb_ref - total_emb_new)
    assert abs(total_emb_ref - total_emb_new) < 1e-5
    postwrap.print_embedding_information()


def test_compute_emb_kernel():
    """Test function to evaluate the xcT second derivatives."""
    # Basic tests
    pot0 = 'mol'
    dm0 = 123
    basis = 'sto-3g'
    mol0 = gto.M(atom="""Ne  0.00000    0.00000    0.00000""",
                 basis=basis)
    emb_args = {'xc_code': 'LDA,VWN', 't_code': 'LDA_K_TF,'}
    pot1 = PyScfEmbPot(mol0, mol0, emb_args)
    dm1 = np.arange(10)
    with pytest.raises(TypeError):
        compute_emb_kernel(pot0, dm0, dm0)
    with pytest.raises(TypeError):
        compute_emb_kernel(pot1, dm0, dm0)
    with pytest.raises(TypeError):
        compute_emb_kernel(pot1, dm1, dm0)
    # Use wrap
    # Compared with ScfWrap results
    mol = Molecule.from_data(mol0.atom)
    mol1 = Molecule.from_data("""He  1.00000    0.00000    0.0000000""")
    method = 'hf'
    args0 = {"mol": mol, "basis": basis, "method": method}
    args1 = {"mol": mol1, "basis": basis, "method": method}
    embs = {"mol": mol, "basis": basis, "method": 'hf',
            "xc_code": 'LDA,VWN', "t_code": 'LDA_K_TF,'}
    wrap = PyScfWrap(args0, args1, embs)
    wrap.run_embedding()
    emb_pot = wrap.pot_object
    dm0 = wrap.vemb_dict["dm0_final"]
    dm1 = wrap.vemb_dict['dm1_ref']
    fxc, ft = compute_emb_kernel(emb_pot, dm0, dm1)
    emb_pot.assign_dm(0, dm0)
    emb_pot.assign_dm(1, dm1)
    ref_vnad = emb_pot.compute_nad_potential()
    exc_nad, et_nad, v_nad_xc, v_nad_t = ref_vnad


if __name__ == "__main__":
    test_embpotbase()
    test_scfwrap()
    test_pyscf_wrap0()
    test_pyscf_wrap_hf_co_h2o_sto3g()
    test_pyscf_wrap_dft_co_h2o_sto3g()
    test_scfwrap_single()
    test_pyscf_wrap_single_co_h2o()
    test_postscfwrap()
    test_postscfwrap_co_h2o()
    test_omolcas_wrap0()
    test_omolcas_wrap_co_h2o_ccpvdz()
    test_compute_emb_kernel()
