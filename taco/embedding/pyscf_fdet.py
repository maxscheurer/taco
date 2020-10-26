#!/usr/bin/env python
# Copyright 2014-2018 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import warnings
import numpy
from pyscf import lib
try:
    from pyscf.dft import libxc
except (ImportError, OSError):
    try:
        from pyscf.dft import xcfun
        libxc = xcfun
    except (ImportError, OSError):
        import warnings
        warnings.warn('XC functional libraries (libxc or XCfun) are not available.')
        from pyscf.dft import xc
        libxc = xc

from pyscf.dft.gen_grid import BLKSIZE
from pyscf import __config__
# Extra imports from numint
from pyscf.dft.numint import _scale_ao, _dot_ao_ao

libdft = lib.load_library('libdft')
OCCDROP = getattr(__config__, 'dft_numint_OCCDROP', 1e-12)
# The system size above which to consider the sparsity of the density matrix.
# If the number of AOs in the system is less than this value, all tensors are
# treated as dense quantities and contracted by dgemm directly.
SWITCH_SIZE = getattr(__config__, 'dft_numint_SWITCH_SIZE', 800)


def eval_mat_emb(mol, ao, weight, rho_both, rho_0, vxc_both, vxc_0,
                 non0tab=None, xctype='LDA', spin=0, verbose=None):
    r'''Calculate XC potential matrix.

    Args:
        mol : an instance of :class:`Mole`

        ao : ([4/10,] ngrids, nao) ndarray
            2D array of shape (N,nao) for LDA,
            3D array of shape (4,N,nao) for GGA
            or (10,N,nao) for meta-GGA.
            N is the number of grids, nao is the number of AO functions.
            If xctype is GGA, ao[0] is AO value and ao[1:3] are the real space
            gradients.  If xctype is meta-GGA, ao[4:10] are second derivatives
            of ao values.
        weight : 1D array
            Integral weights on grids.
        rho : ([4/6,] ngrids) ndarray
            Shape of ((*,N)) for electron density (and derivatives) if spin = 0;
            Shape of ((*,N),(*,N)) for alpha/beta electron density (and derivatives) if spin > 0;
            where N is number of grids.
            rho (*,N) are ordered as (den,grad_x,grad_y,grad_z,laplacian,tau)
            where grad_x = d/dx den, laplacian = \nabla^2 den, tau = 1/2(\nabla f)^2
            In spin unrestricted case,
            rho is ((den_u,grad_xu,grad_yu,grad_zu,laplacian_u,tau_u)
                    (den_d,grad_xd,grad_yd,grad_zd,laplacian_d,tau_d))
        vxc : ([4,] ngrids) ndarray
            XC potential value on each grid = (vrho, vsigma, vlapl, vtau)
            vsigma is GGA potential value on each grid.
            If the kwarg spin != 0, a list [vsigma_uu,vsigma_ud] is required.

    Kwargs:
        xctype : str
            LDA/GGA/mGGA.  It affects the shape of `ao` and `rho`
        non0tab : 2D bool array
            mask array to indicate whether the AO values are zero.  The mask
            array can be obtained by calling :func:`make_mask`
        spin : int
            If not 0, the returned matrix is the Vxc matrix of alpha-spin.  It
            is computed with the spin non-degenerated UKS formula.

    Returns:
        XC potential matrix in 2D array of shape (nao,nao) where nao is the
        number of AO functions.
    '''
    xctype = xctype.upper()
    if xctype == 'LDA' or xctype == 'HF':
        ngrids, nao = ao.shape
    else:
        ngrids, nao = ao[0].shape

    if non0tab is None:
        non0tab = numpy.ones(((ngrids+BLKSIZE-1)//BLKSIZE, mol.nbas),
                             dtype=numpy.uint8)
    shls_slice = (0, mol.nbas)
    ao_loc = mol.ao_loc_nr()
    transpose_for_uks = False
    if xctype == 'LDA' or xctype == 'HF':
        if not isinstance(vxc_both, numpy.ndarray) or vxc_both.ndim == 2:
            vrho = vxc_both[0] - vxc_0[0]
        else:
            vrho = vxc_both - vxc_0
        # *.5 because return mat + mat.T
        # :aow = numpy.einsum('pi,p->pi', ao, .5*weight*vrho)
        aow = _scale_ao(ao, .5*weight*vrho)
        mat = _dot_ao_ao(mol, ao, aow, non0tab, shls_slice, ao_loc)
    else:
        # wv = weight * vsigma * 2
        # aow  = numpy.einsum('pi,p->pi', ao[1], rho[1]*wv)
        # aow += numpy.einsum('pi,p->pi', ao[2], rho[2]*wv)
        # aow += numpy.einsum('pi,p->pi', ao[3], rho[3]*wv)
        # aow += numpy.einsum('pi,p->pi', ao[0], .5*weight*vrho)
        vrho_both, vsigma_both = vxc_both[:2]
        vrho_0, vsigma_0 = vxc_0[:2]
        wv = numpy.empty((4, ngrids))
        if spin == 0:
            assert(vsigma_both is not None and rho_both.ndim==2)
            assert(vsigma_0 is not None and rho_0.ndim==2)
            wv[0] = weight * vrho_both * .5
            wv[0] -= weight * vrho_0 * .5
            wv[1:4] = rho_both[1:4] * (weight * vsigma_both * 2)
            wv[1:4] -= rho_0[1:4] * (weight * vsigma_0 * 2)
        else:
            rho_both_a, rho_both_b = rho_both
            rho_0_a, rho_0_b = rho_0
            wv[0] = weight * vrho_both * .5
            wv[0] -= weight * vrho_0 * .5
            try:
                wv[1:4] = rho_both_a[1:4] * (weight * vsigma_both[0] * 2)  #  sigma_uu
                wv[1:4] += rho_both_b[1:4] * (weight * vsigma_both[1])      #  sigma_ud
                wv[1:4] -= rho_0_a[1:4] * (weight * vsigma_0[0] * 2)  #  sigma_uu
                wv[1:4] -= rho_0_b[1:4] * (weight * vsigma_0[1])      #  sigma_ud
            except ValueError:
                warnings.warn('Note the output of libxc.eval_xc cannot be '
                              'directly used in eval_mat.\nvsigma from eval_xc '
                              'should be restructured as '
                              '(vsigma[:,0],vsigma[:,1])\n')
                transpose_for_uks = True
                vsigma_both = vsigma_both.T
                vsigma_0 = vsigma_0.T
                wv[1:4] = rho_both_a[1:4] * (weight * vsigma_both[0] * 2)  #  sigma_uu
                wv[1:4] += rho_both_b[1:4] * (weight * vsigma_both[1])      #  sigma_ud
                wv[1:4] -= rho_0_a[1:4] * (weight * vsigma_0[0] * 2)  #  sigma_uu
                wv[1:4] -= rho_0_b[1:4] * (weight * vsigma_0[1])      #  sigma_ud
        # :aow = numpy.einsum('npi,np->pi', ao[:4], wv)
        aow = _scale_ao(ao[:4], wv)
        mat = _dot_ao_ao(mol, ao[0], aow, non0tab, shls_slice, ao_loc)

    if xctype == 'MGGA':
        raise NotImplementedError("Only LDA and GGA types are available.")
    return mat + mat.T.conj()
