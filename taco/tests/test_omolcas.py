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

from taco.embedding.emb_pot import EmbPotBase
from taco.embedding.pyscf_emb_pot import PyScfEmbPot
from taco.embedding.scf_wrap import ScfWrap
# from taco.embedding.pyscf_tddft import compute_emb_kernel
from taco.embedding.pyscf_wrap import PyScfWrap
from taco.embedding.postscf_wrap import PostScfWrap
from taco.embedding.omolcas_wrap import OpenMolcasWrap
from taco.testdata.cache import cache


def test_omolcas_wrap_co_h2o_ccpvdz():
    """Test embedded HF-in-HF case."""
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
    read_dm = postwrap.get_density(cache.files["molcas_RUNASCII_co_h2o_cc-pvdz"])
    np.testing.assert_allclose(fin_dm, read_dm, atol=1e-5)
    postwrap.dms_dict['dm0_final'] = read_dm
    postwrap.save_info()
    assert abs(postwrap.energy_dict['et_nad_final'] - embdic['et_nad_final']) < 1e-6
    assert abs(postwrap.energy_dict['exc_nad_final'] - embdic['exc_nad_final']) < 1e-6
    total_emb_ref = embdic['int_emb_xc'] + embdic['int_emb_t']
    total_emb_new = postwrap.energy_dict['int_emb_xc'] + postwrap.energy_dict['int_emb_t']
    # print(total_emb_ref - total_emb_new)
    assert abs(total_emb_ref - total_emb_new) < 1e-5
    postwrap.print_embedding_information()


def test_omolcas_density():
    """Compare densities from RUNFILE/RUNASCII and H5 formats"""
    return True
    


if __name__ == "__main__":
    test_omolcas_wrap_co_h2o_ccpvdz()
