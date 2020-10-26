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
from pyscf.dft.numint import eval_ao, eval_rho, eval_mat

from taco.embedding.pyscf_emb_pot import PyScfEmbPot
from taco.embedding.scf_wrap_single import ScfWrapSingle
from taco.embedding.pyscf_wrap_single import PyScfWrapSingle, get_coulomb_repulsion
from taco.testdata.cache import cache


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


def test_get_coulomb_potential():
    # Check the new function for coulomb repulsion
    # Compared with ScfWrap results
    basis = 'sto-3g'
    mol0 = gto.M(atom="""C        -3.6180905689    1.3768035675   -0.0207958979
                       O        -4.7356838533    1.5255563000    0.1150239130""",
               basis=basis)
    mol1 = gto.M(atom="""O  -7.9563726699    1.4854060709    0.1167920007
                        H  -6.9923165534    1.4211335985    0.1774706091
                        H  -8.1058463545    2.4422204631    0.1115993752""",
                basis=basis)
    nao_mol0 = 10
    nao_mol1 = 7
    ref_dm0 = 2*np.loadtxt(cache.files["co_h2o_sto3g_dma"]).reshape((nao_mol0, nao_mol0))
    ref_dm1 = 2*np.loadtxt(cache.files["co_h2o_sto3g_dmb"]).reshape((nao_mol1, nao_mol1))
    embs = {"xc_code": 'LDA,VWN', "t_code": 'LDA_K_TF,'}
    embpot = PyScfEmbPot(mol0, mol1, embs)
    vemb = embpot.compute_embedding_potential(ref_dm0, ref_dm1)
    v_coul_ref = embpot.vemb_dict['v_coulomb']
    # Construct grid for integration
    grid = gen_grid.Grids(mol0)
    grid.level = 4
    grid.build()
    ao_mol0 = eval_ao(mol0, grid.coords, deriv=0)
    rho0 = eval_rho(mol0, ao_mol0, ref_dm0, xctype='LDA')
    v_coulomb = get_coulomb_repulsion(mol1, ref_dm1, grid.coords)
    v_coul = eval_mat(mol0, ao_mol0, grid.weights, rho0, v_coulomb, xctype='LDA') 
    assert np.allclose(v_coul, v_coul_ref)


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


if __name__ == "__main__":
    test_scfwrap_single()
    test_get_coulomb_potential()
    test_pyscf_wrap_single_co_h2o()
