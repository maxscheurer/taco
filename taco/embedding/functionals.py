"""
"""

import numpy as np

def compute_kinetic_tf(rho):
    """Thomas-Fermi kinetic energy functional."""
    cf = (3./10.)*(np.power(3.0*np.pi**2, 2./3.))
    et = cf*(np.power(rho, 5./3.))
    vt = cf*5./3.*(np.power(rho, 2./3.))
    return et, vt


def compute_exchage_slater(rho):
    """Slater exchange energy functional."""
    from scipy.special import cbrt
    cx = (3./4) * (3/np.pi)**(1./3)
    ex = - cx * (np.power(rho, 4./3))
    vx = -(4./3) * cx * cbrt(np.fabs(rho))
    return ex, vx


def compute_corr_pyscf(rho, xc_code=',VWN'):
    """Correlation energy functional."""
    from pyscf.dft import libxc
    exc, vxc, fxc, kxc = libxc.eval_xc(xc_code, rho)
    return exc, vxc[0]


def compute_kinetic_weizsacker_potential(rho_devs):
    """Compute the Weizsacker Potential.

    Parameters
    ----------
    rho_devs : np.array((6, N))
        Array with the density derivatives,
        density = rho_devs[0]
        grad = rho_devs[1:3] (x, y, z) derivatives
        laplacian = rho_devs[4]
    """
    grad_rho = rho_devs[1:4]
    wpot = 1.0/8.0*(np.einsum('ij,ij->i', grad_rho, grad_rho))/pow(rho_devs[0], 2)
    wpot += - 1.0/4.0*(rho_devs[4])/rho_devs[0]
    return wpot


def compute_kinetic_weizsacker_experiment(rho_devs):
    """Compute the Weizsacker Potential.

    Parameters
    ----------
    rho_devs : np.array((6, N))
        Array with the density derivatives,
        density = rho_devs[0]
        grad = rho_devs[1:3] (x, y, z) derivatives
        laplacian = rho_devs[4]
    """
    zero_mask = np.where(abs(rho_devs[0] - 0.0) > 1e-10)[0]
    wpot = np.zeros(rho_devs[0].shape)
    grad_rho = rho_devs[1:4].T
    wpot[zero_mask] += 1.0/8.0*(np.einsum('ij,ij->i', grad_rho, grad_rho)[zero_mask])
    wpot[zero_mask] /= pow(rho_devs[0][zero_mask], 2)
    wpot[zero_mask] += - 1.0/8.0*(rho_devs[4][zero_mask])/rho_devs[0][zero_mask]
    return wpot


def ndsd_switch_factor(rho_devs, plambda):
    """Compute the NDSD switch factor.

    This formula follows eq. 21 from Garcia-Lastra 2008.

    Paramters
    ---------
    rho_devs : np.array((6, N))
        Array with the density derivatives,
        density = rho_devs[0]
        grad = rho_devs[1:3] (x, y, z) derivatives
        laplacian = rho_devs[4]
    plambda :  float
        Smoothing parameter.
    """
    # TODO: define rhob
    rhob = rho_devs[0]
    sb_min = 0.3
    sb_max = 0.9
    rhob_min = 0.7
    sb = np.linalg.norm(rho_devs[1:4])
    sb /= 2.0*((3*np.pi**2)**(1./3.))*(rhob**(4./3.))
    sfactor = 1.0/(np.exp(plambda*(-sb+sb_min)) + 1.0)
    sfactor *= 1.0/(1.0 - (np.exp(plambda * (-sb + sb_max)) + 1.0))
    sfactor += 1.0/(np.exp(plambda * (-rhob + rhob_min)) + 1.0)
    return sfactor


def compute_kinetic_ndsd_potential(rho0_devs, rho1_devs, plambda):
    """Compute the NDSD energy and potential.

    Parameters
    ----------
    rho0_devs, rho1_devs : np.array((6, N))
        Array with the density derivatives,
        density = rho_devs[0]
        grad = rho_devs[1:3] (x, y, z) derivatives
        laplacian = rho_devs[4]
    """
    rho_tot = rho0_devs[0] + rho1_devs[0]
    etf_tot, vtf_tot = compute_kinetic_tf(rho_tot)
    etf_0, vtf_0 = compute_kinetic_tf(rho0_devs[0])
    etf_1, vtf_1 = compute_kinetic_tf(rho1_devs[0])
    sfactor = ndsd_switch_factor(rho1_devs, plambda)
    wpot = compute_kinetic_weizsacker_experiment(rho1_devs)
    v_ndsd = vtf_tot - vtf_0 + sfactor*wpot
    return v_ndsd


def compute_kinetic_ndsd(rho0_devs, rho1_devs, plambda, grid):
    """Compute the NDSD energy and potential.

    Parameters
    ----------
    rho0_devs, rho1_devs : np.array((6, N))
        Array with the density derivatives,
        density = rho_devs[0]
        grad = rho_devs[1:3] (x, y, z) derivatives
        laplacian = rho_devs[4]
    grid : Grids
        Molecular integration grid from PySCF
    """
    rho_tot = rho0_devs[0] + rho1_devs[0]
    etf_tot, vtf_tot = compute_kinetic_tf(rho_tot)
    etf_0, vtf_0 = compute_kinetic_tf(rho0_devs[0])
    etf_1, vtf_1 = compute_kinetic_tf(rho1_devs[0])
    sfactor = ndsd_switch_factor(rho1_devs, plambda)
 #  wpot = compute_kinetic_weizsacker_potential(rho1_devs)
    wpot = compute_kinetic_weizsacker_experiment(rho1_devs)
    v_ndsd = vtf_tot - vtf_0 + sfactor*wpot
    e_ndsd = etf_tot - etf_0 - etf_1 + np.dot(grid.weigths, rho0_devs[0]*wpot*sfactor)
    return e_ndsd, v_ndsd
