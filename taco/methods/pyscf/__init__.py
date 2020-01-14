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

from pyscf.tdscf import rhf
from pyscf.tdscf.rks import TDRKS
from pyscf.tddft import rks
from pyscf import scf


def FDETDA(mf):
    if isinstance(mf, scf.uhf.UHF):
        raise NotImplementedError('Only Restrited case available')
       #mf = scf.addons.convert_to_uhf(mf)
       #if getattr(mf, 'xc', None):
       #    return uks.TDA(mf)
       #else:
       #    return uhf.TDA(mf)
    else:
        mf = scf.addons.convert_to_rhf(mf)
        if getattr(mf, 'xc', None):
            return rks.FDETDA(mf)
        else:
            raise NotImplementedError('Only DFT case.')
           #return rhf.TDA(mf)

def FDETDDFT(mf):
    if isinstance(mf, scf.uhf.UHF):
        raise NotImplementedError('Only Restrited case available')
       #mf = scf.addons.convert_to_uhf(mf)
       #if getattr(mf, 'xc', None):
       #    if mf._numint.libxc.is_hybrid_xc(mf.xc):
       #        return uks.TDDFT(mf)
       #    else:
       #        return uks.TDDFTNoHybrid(mf)
       #else:
       #    return uhf.TDHF(mf)
    else:
        mf = scf.addons.convert_to_rhf(mf)
        if getattr(mf, 'xc', None):
            if mf._numint.libxc.is_hybrid_xc(mf.xc):
                return rks.FDETDDFT(mf)
            else:
                return rks.FDETDDFTNoHybrid(mf)
        else:
            raise NotImplementedError('Only DFT case.')
           #return rhf.TDHF(mf)

FDETD = FDETDDFT


def FDERPA(mf):
    return FDETDDFT(mf)

def FDEdRPA(mf):
    if isinstance(mf, scf.uhf.UHF):
        raise NotImplementedError('Only Restrited case available')
       #return uks.dRPA(mf)
    else:
        return rks.FDEdRPA(mf)

def FDEdTDA(mf):
    if isinstance(mf, scf.uhf.UHF):
        raise NotImplementedError('Only Restrited case available')
       #return uks.dTDA(mf)
    else:
        return rks.FDEdTDA(mf)
