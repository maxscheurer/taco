"""PySCF Utilities for Embedding calculations."""

import numpy as np
from pyscf.dft import gen_grid
from pyscf.dft.numint import NumInt, nr_rks_fxc_st
from pyscf.dft.numint import eval_ao, eval_rho

from taco.embedding.pyscf_embpot import PyScfEmbPot


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
    grids = gen_grid.Grids(embpot.mol0)
    grids.level = 4
    grids.build()
    mol0 = embpot.mol0
    mol1 = embpot.mol1
    # Evaluate AOs on grid
    ao_mol0 = eval_ao(mol0, grids.coords, deriv=0)
    ao_mol1 = eval_ao(mol1, grids.coords, deriv=0)
    # Get densities on grid
    rho_mol0 = eval_rho(mol0, ao_mol0, dm0, xctype='LDA')
    rho_mol1 = eval_rho(mol1, ao_mol1, dm1, xctype='LDA')
    rho_both = rho_mol0 + rho_mol1
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
