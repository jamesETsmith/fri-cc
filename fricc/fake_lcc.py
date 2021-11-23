import numpy as np

from pyscf.cc import ccsd
from pyscf import lib
from pyscf.lib import logger
from pyscf.cc import rintermediates as imd


class LCCD(ccsd.CCSD):
    def update_amps(self, t1, t2, eris):

        cc = self
        # Ref: Taube and Bartlett J. Chem. Phys. 130, 144112 (2009) Eq. 13 and 15
        assert isinstance(eris, ccsd._ChemistsERIs)
        nocc, nvir = t1.shape
        fock = eris.fock
        mo_e_o = eris.mo_energy[:nocc]
        mo_e_v = eris.mo_energy[nocc:] + cc.level_shift

        foo = fock[:nocc, :nocc].copy()
        fvv = fock[nocc:, nocc:].copy()

        Foo = foo
        Fvv = fvv

        # Move energy terms to the other side
        Foo[np.diag_indices(nocc)] -= mo_e_o
        Fvv[np.diag_indices(nvir)] -= mo_e_v

        # T1 equation
        t1new = np.zeros_like(t1)

        # T2 equation
        t2new = np.zeros_like(t2)
        # Term 1
        t2new += np.asarray(eris.ovov).conj().transpose(0, 2, 1, 3)

        # Term 2
        tmp = lib.einsum("ki,kjab->ijab", Foo, t2)
        t2new -= tmp + tmp.transpose(1, 0, 3, 2)

        # Term 3
        tmp = lib.einsum("ac,ijcb->ijab", Fvv, t2)
        t2new += tmp + tmp.transpose(1, 0, 3, 2)

        Woooo = np.asarray(eris.oooo).transpose(0, 2, 1, 3)
        Wvoov = np.asarray(eris.ovvo).transpose(2, 0, 3, 1)
        Wvovo = np.asarray(eris.oovv).transpose(2, 0, 3, 1)
        Wvvvv = np.asarray(imd._get_vvvv(eris)).transpose(0, 2, 1, 3)

        # Term 4
        t2new += lib.einsum("abcd,ijcd->ijab", Wvvvv, t2)

        # Term 5
        t2new += lib.einsum("klij,klab->ijab", Woooo, t2)

        # Term 6
        tmp = 2 * lib.einsum("akic,kjcb->ijab", Wvoov, t2)
        tmp -= lib.einsum("akci,kjcb->ijab", Wvovo, t2)
        t2new += tmp + tmp.transpose(1, 0, 3, 2)

        tmp = lib.einsum("akic,kjbc->ijab", Wvoov, t2)
        t2new -= tmp + tmp.transpose(1, 0, 3, 2)
        tmp = lib.einsum("bkci,kjac->ijab", Wvovo, t2)
        t2new -= tmp + tmp.transpose(1, 0, 3, 2)

        eia = mo_e_o[:, None] - mo_e_v
        eijab = lib.direct_sum("ia,jb->ijab", eia, eia)
        t1new /= eia
        t2new /= eijab

        return t1new, t2new


class LCCSD(ccsd.CCSD):
    def energy(self, t1=None, t2=None, eris=None):
        '''RCCSD correlation energy'''
        if t1 is None: t1 = self.t1
        if t2 is None: t2 = self.t2
        if eris is None: eris = self.ao2mo()

        nocc, nvir = t1.shape
        fock = eris.fock
        e = 2*np.einsum('ia,ia', fock[:nocc,nocc:], t1)
        # tau = np.einsum('ia,jb->ijab',t1,t1)
        tau = np.zeros_like(t2)
        tau += t2
        eris_ovov = np.asarray(eris.ovov)
        e += 2*np.einsum('ijab,iajb', tau, eris_ovov)
        e +=  -np.einsum('ijab,ibja', tau, eris_ovov)
        if abs(e.imag) > 1e-4:
            logger.warn(self, 'Non-zero imaginary part found in RCCSD energy %s', e)
        return e.real
        
    def update_amps(self, t1, t2, eris):

        cc = self
        # Ref: Taube and Bartlett J. Chem. Phys. 130, 144112 (2009) Eq. 13 and 15
        assert isinstance(eris, ccsd._ChemistsERIs)
        nocc, nvir = t1.shape
        fock = eris.fock
        mo_e_o = eris.mo_energy[:nocc]
        mo_e_v = eris.mo_energy[nocc:] + cc.level_shift

        fov = fock[:nocc, nocc:].copy()
        foo = fock[:nocc, :nocc].copy()
        fvv = fock[nocc:, nocc:].copy()

        Foo = foo
        Fvv = fvv
        Fov = fov

        # Move energy terms to the other side
        Foo[np.diag_indices(nocc)] -= mo_e_o
        Fvv[np.diag_indices(nvir)] -= mo_e_v

        ###############
        # T1 equation #
        ###############
        t1new = np.zeros_like(t1)

        # Term 1
        t1new += fov.conj()
        
        # Term 2
        t1new += np.einsum("ac,ic->ia", Fvv, t1)

        # Term 3
        t1new += -np.einsum("ki,ka->ia", Foo, t1)

        # Term 4
        t1new += 2 * np.einsum("kcai,kc->ia", eris.ovvo, t1)
        t1new += -np.einsum("kiac,kc->ia", eris.oovv, t1)

        # Term 5
        eris_ovvv = np.asarray(eris.get_ovvv())
        t1new += 2 * lib.einsum("kdac,ikcd->ia", eris_ovvv, t2)
        t1new += -lib.einsum("kcad,ikcd->ia", eris_ovvv, t2)

        # Term 6
        eris_ovoo = np.asarray(eris.ovoo, order="C")
        t1new += -2 * lib.einsum("lcki,klac->ia", eris_ovoo, t2)
        t1new += lib.einsum("kcli,klac->ia", eris_ovoo, t2)

        # Term 7
        t1new += 2 * np.einsum("kc,kica->ia", Fov, t2)
        t1new += -np.einsum("kc,ikca->ia", Fov, t2)

        ###############
        # T2 equation #
        ###############
        t2new = np.zeros_like(t2)
        # Term 1
        t2new += np.asarray(eris.ovov).conj().transpose(0, 2, 1, 3)

        # Term 2
        tmp = lib.einsum("ki,kjab->ijab", Foo, t2)
        t2new -= tmp + tmp.transpose(1, 0, 3, 2)

        # Term 3
        tmp = lib.einsum("ac,ijcb->ijab", Fvv, t2)
        t2new += tmp + tmp.transpose(1, 0, 3, 2)

        Woooo = np.asarray(eris.oooo).transpose(0, 2, 1, 3)
        Wvoov = np.asarray(eris.ovvo).transpose(2, 0, 3, 1)
        Wvovo = np.asarray(eris.oovv).transpose(2, 0, 3, 1)
        Wvvvv = np.asarray(imd._get_vvvv(eris)).transpose(0, 2, 1, 3)

        # Term 4
        t2new += lib.einsum("abcd,ijcd->ijab", Wvvvv, t2)

        # Term 5
        t2new += lib.einsum("klij,klab->ijab", Woooo, t2)

        # Term 6
        tmp = 2 * lib.einsum("akic,kjcb->ijab", Wvoov, t2)
        tmp -= lib.einsum("akci,kjcb->ijab", Wvovo, t2)
        t2new += tmp + tmp.transpose(1, 0, 3, 2)
        tmp = lib.einsum("akic,kjbc->ijab", Wvoov, t2)
        t2new -= tmp + tmp.transpose(1, 0, 3, 2)
        tmp = lib.einsum("bkci,kjac->ijab", Wvovo, t2)
        t2new -= tmp + tmp.transpose(1, 0, 3, 2)

        # Term 7
        tmp2 = eris_ovoo.transpose(1, 3, 0, 2).conj()
        tmp = lib.einsum("akij,kb->ijab", tmp2, t1)
        t2new -= tmp + tmp.transpose(1, 0, 3, 2)

        # Term 8
        tmp2 = np.asarray(eris_ovvv).conj().transpose(1, 3, 0, 2)
        tmp = lib.einsum("abic,jc->ijab", tmp2, t1)
        t2new += tmp + tmp.transpose(1, 0, 3, 2)

        eia = mo_e_o[:, None] - mo_e_v
        eijab = lib.direct_sum("ia,jb->ijab", eia, eia)
        t1new /= eia
        t2new /= eijab

        return t1new, t2new
