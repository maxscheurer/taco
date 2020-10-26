"""PySCF Utilities for Embedding calculations."""

import numpy as np
from pyscf import gto
from pyscf.dft import gen_grid
from pyscf.dft.numint import NumInt, nr_rks_fxc_st
from pyscf.dft.numint import eval_ao, eval_rho, eval_mat

from taco.embedding.pyscf_emb_pot import PyScfEmbPot
from taco.testdata.cache import cache


def compute_emb_kernel(embpot, dm0, dm1):
    """Compute the non-additive embedding kernel.

    Parameters
    ----------
    embpot : PyScfEmbPot
        PySCF embeding potential object.
    dm0, dm1 : np.ndarray(NAO,NAO)
        Total density of fragments.
        Restricted case assumed.

    Returns
    -------
    (fxc_emb, ft_emb) : tuple(np.ndarray(NAO, NAO),)
        Contracted embedding non-additive kernel (response).

    """
    if not isinstance(embpot, PyScfEmbPot):
        raise TypeError("`embpot must be a PyScfEmbPot object.")
    if not isinstance(dm0, np.ndarray):
        raise TypeError("`dm0 should be a numpy ndarray.")
    if not isinstance(dm1, np.ndarray):
        raise TypeError("`dm1 should be a numpy ndarray.")
    # Construct grid for integration
    mol0 = embpot.mol0
    mol1 = embpot.mol1
    newatom = '\n'.join([mol0.atom, mol1.atom])
    system = gto.M(atom=newatom, basis=mol0.basis)
    grids = gen_grid.Grids(system)
    grids.level = 4
    grids.build()
    # Evaluate AOs on grid
    ao_mol0 = eval_ao(mol0, grids.coords, deriv=0)
    ao_mol1 = eval_ao(mol1, grids.coords, deriv=0)
    ao_both = eval_ao(system, grids.coords, deriv=0)
    nao_mol0 = mol0.nao_nr()
    nao_mol1 = mol1.nao_nr()
    nao_tot = nao_mol0 + nao_mol1
    # Get densities on grid
    rho_mol0 = eval_rho(mol0, ao_mol0, dm0, xctype='LDA')
    rho_mol1 = eval_rho(mol1, ao_mol1, dm1, xctype='LDA')
    dm_both = np.zeros((nao_tot, nao_tot))
    dm_both[:nao_mol0, :nao_mol0] = dm0
    dm_both[nao_mol0:, nao_mol0:] = dm1
    rho_both = eval_rho(system, ao_both, dm_both, xctype='LDA')
    # rho_both = rho_mol0 + rho_mol1
    # Compute potential and kernel terms
    ni = NumInt()
    xc_code = embpot.emb_args['xc_code']
    t_code = embpot.emb_args['t_code']
    vxc_tot, fxc_tot = ni.eval_xc(xc_code, (rho_both*0.5, rho_both*0.5), 1, deriv=2)[1:3]
    vt_tot, ft_tot = ni.eval_xc(t_code, (rho_both*0.5, rho_both*0.5), 1, deriv=2)[1:3]
    vxc_a, fxc_a = ni.eval_xc(xc_code, (rho_mol0*0.5, rho_mol0*0.5), 1, deriv=2)[1:3]
    vt_a, ft_a = ni.eval_xc(t_code, (rho_mol0*0.5, rho_mol0*0.5), 1, deriv=2)[1:3]
    vxc_nad = (vxc_tot[0] - vxc_a[0], None, None, None)
    vt_nad = (vt_tot[0] - vt_a[0], None, None, None)
    fxc_nad = (fxc_tot[0] - fxc_a[0],) + (None,)*9
    ft_nad = (ft_tot[0] - ft_a[0],) + (None,)*9
    # Get the potential and kernel in the right format
    fxc_emb = nr_rks_fxc_st(ni, mol0, grids, xc_code, dm0, dm0*0.5, 0, True,
                            (rho_mol0*0.5, rho_mol0*0.5), vxc_nad, fxc_nad)
    ft_emb = nr_rks_fxc_st(ni, mol0, grids, t_code, dm0, dm0*0.5, 0, True,
                           (rho_mol0*0.5, rho_mol0*0.5), vt_nad, ft_nad)
    # General case
    vxc_Tot, fxc_Tot = ni.eval_xc(xc_code, rho_both, 0, deriv=2)[1:3]
    vt_Tot, ft_Tot = ni.eval_xc(t_code, rho_both, 0, deriv=2)[1:3]
    vxc_0, fxc_0 = ni.eval_xc(xc_code, rho_mol0, 0, deriv=2)[1:3]
    vt_0, ft_0 = ni.eval_xc(t_code, rho_mol0, 0, deriv=2)[1:3]
    print('Exchange-correlation part')
    print(fxc_Tot[0])
    print(fxc_0[0])
    print(fxc_Tot[0] - fxc_0[0])
    print('Kinetic part')
    print(ft_Tot[0])
    print(ft_0[0])
    print(ft_Tot[0] - ft_0[0])
    vxc = (vxc_Tot[0] - vxc_0[0], None, None, None)
    vts = (vt_Tot[0] - vt_0[0], None, None, None)
    fxc = (fxc_Tot[0] - fxc_0[0],) + (None,)*9
    fts = (ft_Tot[0] - ft_0[0],) + (None,)*9
    v1xc = ni.nr_rks_fxc(mol0, grids, xc_code, dm0, dm0, 0, True,
                         rho_mol0, vxc, fxc)
    v1ts = ni.nr_rks_fxc(mol0, grids, t_code, dm0, dm0, 0, True,
                         rho_mol0, vts, fts)
    # return (fxc_emb, ft_emb)
    return (v1xc, v1ts)


def finite_difference_fxc(xc_code, rho):
    """Compute finite difference kernel"""
    ni = NumInt()
    fxc = np.zeros(rho.shape)
    delta_rho = 1e-6
    rho2 = np.copy(rho)
    rho2 += delta_rho  # rho + delta_rho
    vc1 = ni.eval_xc(xc_code, rho2, 0)[1]
    if (rho < 2.0*delta_rho).all:  # use forward difference
        vc2 = ni.eval_xc(xc_code, rho, 0)[1]
        fxc = (vc1[0] - vc2[0])/(delta_rho)
    else:  # centered difference
        rho2 -= 2.0*delta_rho  # rho - delta_rho
        vc2 = ni.eval_xc(xc_code, rho2, 0)[1]
        fxc = (vc1[0] - vc2[0])/(2.0*delta_rho)
    return fxc


if __name__ == "__main__":
    # Check second order derivative with respect to density
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
    xc_code = 'LDA,VWN'
    t_code = 'LDA_K_TF,'
    dm0 = 2*np.loadtxt(cache.files["co_h2o_sto3g_dma"]).reshape((nao_co, nao_co))
    dm1 = 2*np.loadtxt(cache.files["co_h2o_sto3g_dmb"]).reshape((nao_h2o, nao_h2o))
    embs = {"xc_code": xc_code, "t_code": t_code}
    embpot = PyScfEmbPot(co, h2o, embs)
    grids = gen_grid.Grids(co)
    grids.level = 4
    grids.build()
    # Evaluate AOs on grid
    ao_mol0 = eval_ao(co, grids.coords, deriv=0)
    ao_mol1 = eval_ao(h2o, grids.coords, deriv=0)
    # Get densities on grid
    rho_mol0 = eval_rho(co, ao_mol0, dm0, xctype='LDA')
    rho_mol1 = eval_rho(h2o, ao_mol1, dm1, xctype='LDA')
    ni = NumInt()
    vxc_0, fxc_0 = ni.eval_xc(xc_code, rho_mol0, 0, deriv=2)[1:3]
    vt_0, ft_0 = ni.eval_xc(t_code, rho_mol0, 0, deriv=2)[1:3]
    fxc_again = finite_difference_fxc(xc_code, rho_mol0)
    print(fxc_again)
    print(fxc_0[0])
    #assert np.allclose(fxc_again, fxc_0[0], rtol=1e-4)
    vxc = (vxc_0[0],) + (None,)*3
    fxc = (fxc_0[0],) + (None,)*9
    fxc2 = (fxc_again,) + (None,)*9
    v1xc = ni.nr_rks_fxc(co, grids, xc_code, dm0, dm0, 0, True,
                         rho_mol0, vxc, fxc)
    v2xc = ni.nr_rks_fxc(co, grids, xc_code, dm0, dm0, 0, True,
                         rho_mol0, vxc, fxc2)
    v3xc = eval_mat(co, ao_mol1, grids.weights, rho_mol0, fxc[0]*rho_mol0, xctype='LDA')
    print("Contracted kernel")
    print(v1xc)
    print("Other")
    print(v2xc)
    print("one more!!")
    print(v3xc)
    print(2.0*np.einsum('ab,ba', v1xc, dm0))
    print(2.0*np.einsum('ab,ba', v2xc, dm0))

    # fxc, ft = compute_emb_kernel(emb_pot, dm0, dm1)
