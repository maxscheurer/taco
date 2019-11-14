#!/usr/bin/env python
""" FDET Restricted TDDFT Classes for pyscf."""

from functools import reduce
import numpy
from pyscf import lib
from pyscf.lib import logger
from pyscf import symm
from pyscf.dft import numint
from pyscf.tdscf import rks
from pyscf.scf import rohf, uhf
from pyscf.scf import hf_symm
from pyscf.ao2mo import _ao2mo
from pyscf.data import nist
from pyscf import __config__


# Low excitation filter to avoid numerical instability
POSTIVE_EIG_THRESHOLD = getattr(__config__, 'tdscf_rhf_TDDFT_positive_eig_threshold', 1e-3)


def _gen_fde_rhf_response(mf, vxc_emb, fxc_emb, mo_coeff=None, mo_occ=None,
                          singlet=None, hermi=0, max_memory=None):
    assert(not isinstance(mf, (uhf.UHF, rohf.ROHF)))

    if mo_coeff is None: mo_coeff = mf.mo_coeff
    if mo_occ is None: mo_occ = mf.mo_occ
    mol = mf.mol
    if getattr(mf, 'xc', None) and getattr(mf, '_numint', None):
        from pyscf.dft import rks
        ni = mf._numint
        ni.libxc.test_deriv_order(mf.xc, 2, raise_error=True)
        if getattr(mf, 'nlc', '') != '':
            logger.warn(mf, 'NLC functional found in DFT object.  Its second '
                        'deriviative is not available. Its contribution is '
                        'not included in the response function.')
        omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, mol.spin)
        hybrid = abs(hyb) > 1e-10

        # mf can be pbc.dft.RKS object with multigrid
        if (not hybrid and
          'MultiGridFFTDF' == getattr(mf, 'with_df', None).__class__.__name__):
            from pyscf.pbc.dft import multigrid
            dm0 = mf.make_rdm1(mo_coeff, mo_occ)
            return multigrid._gen_rhf_response(mf, dm0, singlet, hermi)

        if singlet is None:  # for newton solver
            rho0, vxc, fxc = ni.cache_xc_kernel(mol, mf.grids, mf.xc,
                                                mo_coeff, mo_occ, 0)
        else:
            rho0, vxc, fxc = ni.cache_xc_kernel(mol, mf.grids, mf.xc,
                                                [mo_coeff]*2, [mo_occ*.5]*2, spin=1)

        dm0 = None # mf.make_rdm1(mo_coeff, mo_occ)

        if max_memory is None:
            mem_now = lib.current_memory()[0]
            max_memory = max(2000, mf.max_memory*.8-mem_now)

        if singlet is None:  # Without specify singlet, general case
            def vind(dm1):
                # The singlet hessian
                if hermi == 2:
                    v1 = numpy.zeros_like(dm1)
                else:
                    v1 = ni.nr_rks_fxc(mol, mf.grids, mf.xc, dm0, dm1, 0, hermi,
                                       rho0, vxc, fxc, max_memory=max_memory)
                if hybrid:
                    if hermi != 2:
                        vj, vk = mf.get_jk(mol, dm1, hermi=hermi)
                        vk *= hyb
                        if abs(omega) > 1e-10:  # For range separated Coulomb
                            vk += rks._get_k_lr(mol, dm1, omega, hermi) * (alpha-hyb)
                        v1 += vj - .5 * vk
                    else:
                        v1 -= .5 * hyb * mf.get_k(mol, dm1, hermi=hermi)
                elif hermi != 2:
                    v1 += mf.get_j(mol, dm1, hermi=hermi)
                return v1

        elif singlet:
            def vind(dm1):
                if hermi == 2:
                    v1 = numpy.zeros_like(dm1)
                else:
                    # nr_rks_fxc_st requires alpha of dm1, dm1*.5 should be scaled
                    v1 = numint.nr_rks_fxc_st(ni, mol, mf.grids, mf.xc, dm0, dm1, 0,
                                              True, rho0, vxc, fxc,
                                              max_memory=max_memory)
                    v1 *= .5
                if hybrid:
                    if hermi != 2:
                        vj, vk = mf.get_jk(mol, dm1, hermi=hermi)
                        vk *= hyb
                        if abs(omega) > 1e-10:  # For range separated Coulomb
                            vk += rks._get_k_lr(mol, dm1, omega, hermi) * (alpha-hyb)
                        v1 += vj - .5 * vk
                    else:
                        v1 -= .5 * hyb * mf.get_k(mol, dm1, hermi=hermi)
                elif hermi != 2:
                    v1 += mf.get_j(mol, dm1, hermi=hermi)
                return v1
        else:  # triplet
            def vind(dm1):
                if hermi == 2:
                    v1 = numpy.zeros_like(dm1)
                else:
                    # nr_rks_fxc_st requires alpha of dm1, dm1*.5 should be scaled
                    v1 = numint.nr_rks_fxc_st(ni, mol, mf.grids, mf.xc, dm0, dm1, 0,
                                              False, rho0, vxc, fxc,
                                              max_memory=max_memory)
                    v1 *= .5
                if hybrid:
                    vk = mf.get_k(mol, dm1, hermi=hermi)
                    vk *= hyb
                    if abs(omega) > 1e-10:  # For range separated Coulomb
                        vk += rks._get_k_lr(mol, dm1, omega, hermi) * (alpha-hyb)
                    v1 += -.5 * vk
                return v1

    else:  # HF
        if (singlet is None or singlet) and hermi != 2:
            def vind(dm1):
                vj, vk = mf.get_jk(mol, dm1, hermi=hermi)
                return vj - .5 * vk
        else:
            def vind(dm1):
                return -.5 * mf.get_k(mol, dm1, hermi=hermi)

    return vind


def gen_tda_operation(mf, vxc_emb, fxc_emb, fock_ao=None, singlet=True, wfnsym=None):
    '''Generate function to compute (A+B)x

    Kwargs:
        wfnsym : int or str
            Point group symmetry irrep symbol or ID for excited CIS wavefunction.
    '''
    mol = mf.mol
    mo_coeff = mf.mo_coeff
    assert(mo_coeff.dtype == numpy.double)
    mo_energy = mf.mo_energy
    mo_occ = mf.mo_occ
    nao, nmo = mo_coeff.shape
    occidx = numpy.where(mo_occ == 2)[0]
    viridx = numpy.where(mo_occ == 0)[0]
    nocc = len(occidx)
    nvir = len(viridx)
    orbv = mo_coeff[:, viridx]
    orbo = mo_coeff[:, occidx]

    if wfnsym is not None and mol.symmetry:
        if isinstance(wfnsym, str):
            wfnsym = symm.irrep_name2id(mol.groupname, wfnsym)
        wfnsym = wfnsym % 10  # convert to D2h subgroup
        orbsym = hf_symm.get_orbsym(mol, mo_coeff) % 10
        sym_forbid = (orbsym[occidx, None] ^ orbsym[viridx]) != wfnsym

    if fock_ao is None:
        # dm0 = mf.make_rdm1(mo_coeff, mo_occ)
        # fock_ao = mf.get_hcore() + mf.get_veff(mol, dm0)
        foo = numpy.diag(mo_energy[occidx])
        fvv = numpy.diag(mo_energy[viridx])
    else:
        fock = reduce(numpy.dot, (mo_coeff.conj().T, fock_ao, mo_coeff))
        foo = fock[occidx[:, None], occidx]
        fvv = fock[viridx[:, None], viridx]

    hdiag = (fvv.diagonal().reshape(-1, 1) - foo.diagonal()).T
    if wfnsym is not None and mol.symmetry:
        hdiag[sym_forbid] = 0
    hdiag = hdiag.ravel()

    mo_coeff = numpy.asarray(numpy.hstack((orbo, orbv)), order='F')
    vresp = _gen_fde_rhf_response(mf, vxc_emb, fxc_emb, singlet=singlet, hermi=0)

    def vind(zs):
        nz = len(zs)
        if wfnsym is not None and mol.symmetry:
            zs = numpy.copy(zs).reshape(-1, nocc, nvir)
            zs[:, sym_forbid] = 0
        dmov = numpy.empty((nz, nao, nao))
        for i, z in enumerate(zs):
            # *2 for double occupancy
            dmov[i] = reduce(numpy.dot, (orbo, z.reshape(nocc, nvir)*2, orbv.conj().T))
        v1ao = vresp(dmov)
        # v1ov = numpy.asarray([reduce(numpy.dot, (orbo.T, v, orbv)) for v in v1ao])
        v1ov = _ao2mo.nr_e2(v1ao, mo_coeff, (0, nocc, nocc, nmo)).reshape(-1, nocc, nvir)
        for i, z in enumerate(zs):
            v1ov[i] += numpy.einsum('sp,qs->qp', fvv, z.reshape(nocc, nvir))
            v1ov[i] -= numpy.einsum('sp,pr->sr', foo, z.reshape(nocc, nvir))
        if wfnsym is not None and mol.symmetry:
            v1ov[:, sym_forbid] = 0
        return v1ov.reshape(nz, -1)

    return vind, hdiag


gen_tda_hop = gen_tda_operation


class FDETDA(rks.TDA):
    def __init__(self, mf, vxc_emb, fxc_emb):
        self.vxc_emb = vxc_emb
        self.fxc_emb = fxc_emb
        rks.TDA.__init__(self, mf)

    def gen_vind(self, mf):
        '''Compute Ax'''
        return gen_tda_hop(mf, self.vxc_emb, self.fxc_emb, singlet=self.singlet, wfnsym=self.wfnsym)


def gen_tdhf_operation(mf, vxc_emb, fxc_emb, fock_ao=None, singlet=True, wfnsym=None):
    '''Generate function to compute

    [ A  B][X]
    [-B -A][Y]
    '''
    mol = mf.mol
    mo_coeff = mf.mo_coeff
    assert(mo_coeff.dtype == numpy.double)
    mo_energy = mf.mo_energy
    mo_occ = mf.mo_occ
    nao, nmo = mo_coeff.shape
    occidx = numpy.where(mo_occ == 2)[0]
    viridx = numpy.where(mo_occ == 0)[0]
    nocc = len(occidx)
    nvir = len(viridx)
    orbv = mo_coeff[:, viridx]
    orbo = mo_coeff[:, occidx]

    if wfnsym is not None and mol.symmetry:
        if isinstance(wfnsym, str):
            wfnsym = symm.irrep_name2id(mol.groupname, wfnsym)
        wfnsym = wfnsym % 10  # convert to D2h subgroup
        orbsym = hf_symm.get_orbsym(mol, mo_coeff) % 10
        sym_forbid = (orbsym[occidx, None] ^ orbsym[viridx]) != wfnsym

    foo = numpy.diag(mo_energy[occidx])
    fvv = numpy.diag(mo_energy[viridx])

    hdiag = (fvv.diagonal().reshape(-1, 1) - foo.diagonal()).T
    if wfnsym is not None and mol.symmetry:
        hdiag[sym_forbid] = 0
    hdiag = numpy.hstack((hdiag.ravel(), hdiag.ravel()))

    mo_coeff = numpy.asarray(numpy.hstack((orbo, orbv)), order='F')
    vresp = _gen_fde_rhf_response(mf, vxc_emb, fxc_emb, singlet=singlet, hermi=0)

    def vind(xys):
        nz = len(xys)
        if wfnsym is not None and mol.symmetry:
            # shape(nz,2,nocc,nvir): 2 ~ X,Y
            xys = numpy.copy(xys).reshape(nz, 2, nocc, nvir)
            xys[:, :, sym_forbid] = 0
        dms = numpy.empty((nz, nao, nao))
        for i in range(nz):
            x, y = xys[i].reshape(2, nocc, nvir)
            # *2 for double occupancy
            dmx = reduce(numpy.dot, (orbo, x*2, orbv.T))
            dmy = reduce(numpy.dot, (orbv, y.T*2, orbo.T))
            dms[i] = dmx + dmy  # AX + BY

        v1ao = vresp(dms)
        v1ov = _ao2mo.nr_e2(v1ao, mo_coeff, (0, nocc, nocc, nmo)).reshape(-1, nocc, nvir)
        v1vo = _ao2mo.nr_e2(v1ao, mo_coeff, (nocc, nmo, 0, nocc)).reshape(-1, nvir, nocc)
        hx = numpy.empty((nz, 2, nocc, nvir), dtype=v1ov.dtype)
        for i in range(nz):
            x, y = xys[i].reshape(2, nocc, nvir)
            hx[i, 0] = v1ov[i]
            hx[i, 0] += numpy.einsum('sp,qs->qp', fvv, x)  # AX
            hx[i, 0] -= numpy.einsum('sp,pr->sr', foo, x)  # AX
            hx[i, 1] =-v1vo[i].T
            hx[i, 1] -= numpy.einsum('sp,qs->qp', fvv, y)  # -AY
            hx[i, 1] += numpy.einsum('sp,pr->sr', foo, y)  # -AY

        if wfnsym is not None and mol.symmetry:
            hx[:, :, sym_forbid] = 0
        return hx.reshape(nz, -1)

    return vind, hdiag


class FDETDDFT(rks.TDHF):
    def __init__(self, mf, vxc_emb, fxc_emb):
        self.vxc_emb = vxc_emb
        self.fxc_emb = fxc_emb
        rks.TDHF.__init__(self, mf)

    @lib.with_doc(gen_tdhf_operation.__doc__)
    def gen_vind(self, mf):
        return gen_tdhf_operation(mf, self.vxc_emb, self.fxc_emb, singlet=self.singlet,
                                  wfnsym=self.wfnsym)


FDERPA = FDETDRKS = FDETDDFT


class FDETDDFTNoHybrid(FDETDA):
    ''' Solve (A-B)(A+B)(X+Y) = (X+Y)w^2
    '''
    def gen_vind(self, mf):
        wfnsym = self.wfnsym
        singlet = self.singlet

        mol = mf.mol
        mo_coeff = mf.mo_coeff
        assert(mo_coeff.dtype == numpy.double)
        mo_energy = mf.mo_energy
        mo_occ = mf.mo_occ
        nao, nmo = mo_coeff.shape
        occidx = numpy.where(mo_occ == 2)[0]
        viridx = numpy.where(mo_occ == 0)[0]
        nocc = len(occidx)
        nvir = len(viridx)
        orbv = mo_coeff[:, viridx]
        orbo = mo_coeff[:, occidx]

        if wfnsym is not None and mol.symmetry:
            if isinstance(wfnsym, str):
                wfnsym = symm.irrep_name2id(mol.groupname, wfnsym)
            wfnsym = wfnsym % 10  # convert to D2h subgroup
            orbsym = hf_symm.get_orbsym(mol, mo_coeff) % 10
            sym_forbid = (orbsym[occidx, None] ^ orbsym[viridx]) != wfnsym

        e_ia = (mo_energy[viridx].reshape(-1, 1) - mo_energy[occidx]).T
        if wfnsym is not None and mol.symmetry:
            e_ia[sym_forbid] = 0
        d_ia = numpy.sqrt(e_ia).ravel()
        ed_ia = e_ia.ravel() * d_ia
        hdiag = e_ia.ravel() ** 2

        vresp = _gen_fde_rhf_response(mf, self.vxc_emb, self.fxc_emb, singlet=singlet, hermi=1)

        def vind(zs):
            nz = len(zs)
            dmov = numpy.empty((nz, nao, nao))
            for i, z in enumerate(zs):
                # *2 for double occupancy
                dm = reduce(numpy.dot, (orbo, (d_ia*z).reshape(nocc, nvir)*2, orbv.T))
                dmov[i] = dm + dm.T # +cc for A+B and K_{ai,jb} in A == K_{ai,bj} in B
            v1ao = vresp(dmov)
            v1ov = _ao2mo.nr_e2(v1ao, mo_coeff, (0, nocc, nocc, nmo)).reshape(-1, nocc*nvir)
            for i, z in enumerate(zs):
                # numpy.sqrt(e_ia) * (e_ia*d_ia*z + v1ov)
                v1ov[i] += ed_ia * z
                v1ov[i] *= d_ia
            return v1ov.reshape(nz, -1)

        return vind, hdiag

    def kernel(self, x0=None, nstates=None):
        '''TDDFT diagonalization solver
        '''
        mf = self._scf
        if mf._numint.libxc.is_hybrid_xc(mf.xc):
            raise RuntimeError('%s cannot be used with hybrid functional'
                               % self.__class__)
        self.check_sanity()
        self.dump_flags()
        if nstates is None:
            nstates = self.nstates
        else:
            self.nstates = nstates

        log = lib.logger.Logger(self.stdout, self.verbose)

        vind, hdiag = self.gen_vind(self._scf)
        precond = self.get_precond(hdiag)
        if x0 is None:
            x0 = self.init_guess(self._scf, self.nstates)

        def pickeig(w, v, nroots, envs):
            idx = numpy.where(w > POSTIVE_EIG_THRESHOLD**2)[0]
            return w[idx], v[:, idx], idx

        self.converged, w2, x1 = \
                lib.davidson1(vind, x0, precond,
                              tol=self.conv_tol,
                              nroots=nstates, lindep=self.lindep,
                              max_space=self.max_space, pick=pickeig,
                              verbose=log)

        mo_energy = self._scf.mo_energy
        mo_occ = self._scf.mo_occ
        occidx = numpy.where(mo_occ == 2)[0]
        viridx = numpy.where(mo_occ == 0)[0]
        e_ia = (mo_energy[viridx, None] - mo_energy[occidx]).T
        e_ia = numpy.sqrt(e_ia)

        def norm_xy(w, z):
            zp = e_ia * z.reshape(e_ia.shape)
            zm = w/e_ia * z.reshape(e_ia.shape)
            x = (zp + zm) * .5
            y = (zp - zm) * .5
            norm = lib.norm(x)**2 - lib.norm(y)**2
            norm = numpy.sqrt(.5/norm)  # normalize to 0.5 for alpha spin
            return (x*norm, y*norm)

        idx = numpy.where(w2 > POSTIVE_EIG_THRESHOLD**2)[0]
        self.e = numpy.sqrt(w2[idx])
        self.xy = [norm_xy(self.e[i], x1[i]) for i in idx]

        if self.chkfile:
            lib.chkfile.save(self.chkfile, 'tddft/e', self.e)
            lib.chkfile.save(self.chkfile, 'tddft/xy', self.xy)

        log.note('Excited State energies (eV)\n%s', self.e * nist.HARTREE2EV)
        return self.e, self.xy

    def nuc_grad_method(self):
        from pyscf.grad import tdrks
        return tdrks.Gradients(self)


class FDEdRPA(FDETDDFTNoHybrid):
    def __init__(self, mf, vxc_emb, fxc_emb):
        if not getattr(mf, 'xc', None):
            raise RuntimeError("direct RPA can only be applied with DFT; for HF+dRPA, use .xc='hf'")
        from pyscf import scf
        mf = scf.addons.convert_to_rhf(mf)
        mf.xc = ''
        FDETDDFTNoHybrid.__init__(self, mf, vxc_emb, fxc_emb)


FDETDH = FDEdRPA


class FDEdTDA(FDETDA):
    def __init__(self, mf, vxc_emb, fxc_emb):
        if not getattr(mf, 'xc', None):
            raise RuntimeError("direct TDA can only be applied with DFT; for HF+dTDA, use .xc='hf'")
        from pyscf import scf
        mf = scf.addons.convert_to_rhf(mf)
        mf.xc = ''
        FDETDA.__init__(self, mf, vxc_emb, fxc_emb)


from pyscf import dft

dft.rks.RKS.FDETDA           = dft.rks_symm.RKS.FDETDA           = lib.class_as_method(FDETDA)
dft.rks.RKS.FDETDDFT         = dft.rks_symm.RKS.FDETDDFT         = lib.class_as_method(FDETDDFT)
dft.rks.RKS.FDETDDFTNoHybrid = dft.rks_symm.RKS.FDETDDFTNoHybrid = lib.class_as_method(FDETDDFTNoHybrid)
dft.rks.RKS.FDEdTDA          = dft.rks_symm.RKS.FDEdTDA          = lib.class_as_method(FDEdTDA)
dft.rks.RKS.FDEdRPA          = dft.rks_symm.RKS.FDEdRPA          = lib.class_as_method(FDEdRPA)
