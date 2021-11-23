import time
import numpy as np
import copy

from pyscf.cc import ccsd
from pyscf import lib
from pyscf.lib import logger
from pyscf.cc import rintermediates as imd
from emcee.autocorr import integrated_time

from .py_rccsd import SparseTensor4d
from .py_rccsd import contract_DTSpT
from .friccsd import FRICCSD, default_fri_settings, ALLOWED_CONTRACTIONS

numpy = np
default_fri_settings["LCCD"] = False

# fmt: off
def update_amps(
    cc,
    t1: np.ndarray,
    t2: np.ndarray,
    eris: ccsd._ChemistsERIs,
):
    """Update the CC t1 and t2 amplitudes using a compressed t2 tensor for
    several of the contractions.

    Parameters
    ----------
    cc : fricc.FRILCCSD
        CC object
    t1 : np.ndarray
        t1 amplitudes (nocc,nvirt)
    t2 : np.ndarray
        t2 amplitudes (nocc,nocc,nvirt,nvirt)
    eris : ccsd._ChemistsERIs
        2e-integrals

    Returns
    -------
    (np.ndarray,np.ndarray)
        The new t1 and t2 amplitudes

    Notes
    -----
    Ref: Taube and Bartlett J. Chem. Phys. 130, 144112 (2009) Eq. 13 and 15

    """
    #
    # Check Args
    #

    assert isinstance(eris, ccsd._ChemistsERIs)

    log = lib.logger.new_logger(cc)
    m_keep = cc.fri_settings["m_keep"]
    # TODO add input checks

    m_keep = cc.fri_settings["m_keep"]
    if m_keep > t2.size or m_keep == 0:
        raise ValueError(
            "Bad m_keep value! The following condition wasn't met 0 < m_keep <= t2.size"
        )

    compressed_contractions = cc.fri_settings["compressed_contractions"]
    contraction_timings = {}

    #
    # Shorthands
    #
    nocc, nvir = t1.shape
    fock = eris.fock
    mo_e_o = eris.mo_energy[:nocc]
    mo_e_v = eris.mo_energy[nocc:] + cc.level_shift

    fov = fock[:nocc, nocc:].copy()
    t_update_amps = time.perf_counter()

    #
    # Compression
    #
    log.debug(f"M_KEEP = {m_keep} of {t2.size}")
    # log.warn(f"|t2|_1 = {np.linalg.norm(t2.ravel(), ord=1)}")
    # np.save("fricc_t2.npy", t2.ravel())
    # exit(0)

    t_compress = time.perf_counter()
    t2_sparse = SparseTensor4d(
        t2,
        t2.shape,
        int(m_keep),
        cc.fri_settings["compression"],
        cc.fri_settings["sampling_method"],
        cc.fri_settings["verbose"],
    )
    t_compress = time.perf_counter() - t_compress

    #
    # Updating the Amplitudes
    #

    Foo = fock[:nocc, :nocc].copy()
    Fvv = fock[nocc:, nocc:].copy()

    # Move energy terms to the other side
    Foo[np.diag_indices(nocc)] -= mo_e_o
    Fvv[np.diag_indices(nvir)] -= mo_e_v

    # From So Hirata http://faculty.scs.illinois.edu/hirata/lccsd_t1.out
    # ✓ [ + 1.0 ] * f ( p2 h1 ) 
    # ✓ [ - 1.0 ] * Sum ( h3 ) * f ( h3 h1 ) * t ( p2 h3 )
    # ✓ [ + 1.0 ] * Sum ( p3 ) * f ( p2 p3 ) * t ( p3 h1 )

    #   [ - 1.0 ] * Sum ( h4 p3 ) * t ( p3 h4 ) * v ( h4 p2 h1 p3 )
    # ✓ [ + 1.0 ] * Sum ( h3 p4 ) * f ( h3 p4 ) * t ( p4 p2 h3 h1 )

    # ✓ [ + 0.5 ] * Sum ( h4 h5 p3 ) * t ( p3 p2 h4 h5 ) * v ( h4 h5 h1 p3 )
    # ✓ [ + 0.5 ] * Sum ( h5 p3 p4 ) * t ( p3 p4 h5 h1 ) * v ( h5 p2 p3 p4 )

    # T1 equation 13 in Bartlett
    t1new = np.zeros_like(t1)

    # Term 1
    t1new += fov.conj()
    # print(f"Term 1 {np.linalg.norm(fov.conj()):3e}")

    # Term 2
    t1new += np.einsum("ac,ic->ia", Fvv, t1)
    # _tmp = np.einsum("ac,ic->ia", Fvv, t1)
    # print(f"Term 2 {np.linalg.norm(_tmp):3e}")

    # Term 3
    t1new -= np.einsum("ki,ka->ia", Foo, t1)
    # _tmp = np.einsum("ki,ka->ia", Foo, t1)
    # print(f"Term 3 {np.linalg.norm(_tmp):3e}")


    # Term 4
    # TODO: INCONSISTENT between Stanton and Taube paper (and Hirata's website)
    t1new += (2 * np.einsum('kcai,kc->ia', eris.ovvo, t1) - np.einsum('kiac,kc->ia', eris.oovv, t1))
    _tmp = (2 * np.einsum("kcai,kc->ia", eris.ovvo, t1) - np.einsum("kiac,kc->ia", eris.oovv, t1))
    print(f"Term 4 {np.linalg.norm(_tmp):3e}")


    # Term 5
    eris_ovvv = np.asarray(eris.get_ovvv())
    t1new += (2 * lib.einsum("kdac,ikcd->ia", eris_ovvv, t2) - lib.einsum("kcad,ikcd->ia", eris_ovvv, t2))
    # _tmp = (2 * lib.einsum("kdac,ikcd->ia", eris_ovvv, t2) - lib.einsum("kcad,ikcd->ia", eris_ovvv, t2))
    # print(f"Term 5 {np.linalg.norm(_tmp):3e}")

    # Term 6
    eris_ovoo = np.asarray(eris.ovoo, order="C")
    t1new -= (2 * lib.einsum("lcki,klac->ia", eris_ovoo, t2) - lib.einsum("kcli,klac->ia", eris_ovoo, t2))
    # _tmp = (2 * lib.einsum("lcki,klac->ia", eris_ovoo, t2) - lib.einsum("kcli,klac->ia", eris_ovoo, t2))
    # print(f"Term 6 {np.linalg.norm(_tmp):3e}")

    # LinΛCCSD Term in Eq 14
    # t1new += np.einsum("jiab,jb->ia", t2, fov)
    # _tmp = 2*np.einsum('kc,kica->ia', fov, t2)  -np.einsum('kc,ikca->ia', fov, t2)
    # t1new += _tmp
    # print(np.linalg.norm(_tmp))

    # LCCD
    if cc.fri_settings["LCCD"]:
        t1 = np.zeros_like(t1)
        t1new = np.zeros_like(t1) # DO LCCD

    #
    # T2 equation
    #

    # Term 1
    t2new = np.zeros_like(t2)
    t2new += np.asarray(eris.ovov).conj().transpose(0, 2, 1, 3)

    # Term 2
    tmp = lib.einsum("ki,kjab->ijab", Foo, t2)
    t2new -= tmp + tmp.transpose(1, 0, 3, 2)

    # Term 3
    tmp = lib.einsum("ac,ijcb->ijab", Fvv, t2)
    t2new += (tmp + tmp.transpose(1, 0, 3, 2))

    Woooo = np.asarray(eris.oooo).copy().transpose(0, 2, 1, 3)
    Wvoov = np.asarray(eris.ovvo).copy().transpose(2, 0, 3, 1)
    Wvovo = np.asarray(eris.oovv).copy().transpose(2, 0, 3, 1)
    Wvvvv = np.asarray(imd._get_vvvv(eris)).transpose(0, 2, 1, 3)

    # Term 4
    if "O^2V^4" in compressed_contractions:
        t_2323 = time.perf_counter()
        contract_DTSpT(Wvvvv, t2_sparse, t2new, "2323")
        contraction_timings["2323"] = time.perf_counter() - t_2323
    else:
        t2new += lib.einsum("abcd,ijcd->ijab", Wvvvv, t2)

    # Term 5
    if "O^4V^2" in compressed_contractions:
        t_0101 = time.perf_counter()
        contract_DTSpT(Woooo, t2_sparse, t2new, "0101")
        contraction_timings["0101"] = time.perf_counter() - t_0101
    else:
        tau = t2  # + np.einsum("ia,jb->ijab", t1, t1)
        t2new += lib.einsum("klij,klab->ijab", Woooo, tau)


    # FRI-Compressed contraction
    # Term 6
    if "O^3V^3" in compressed_contractions:
        tmp = np.zeros_like(t2new)
        t_1302 = time.perf_counter()
        contract_DTSpT(Wvoov, t2_sparse, tmp, "1302")
        contraction_timings["1302"] = time.perf_counter() - t_1302

        t_1202 = time.perf_counter()
        contract_DTSpT(Wvovo, t2_sparse, tmp, "1202")
        contraction_timings["1202"] = time.perf_counter() - t_1202

        t2new += tmp + tmp.transpose(1, 0, 3, 2)

        tmp = np.zeros_like(t2new)
        t_1303 = time.perf_counter()
        contract_DTSpT(Wvoov, t2_sparse, tmp, "1303")
        contraction_timings["1303"] = time.perf_counter() - t_1303

        t2new -= tmp + tmp.transpose(1, 0, 3, 2)

        tmp = np.zeros_like(t2new)
        t_1203 = time.perf_counter()
        contract_DTSpT(Wvovo, t2_sparse, tmp, "1203")
        contraction_timings["1203"] = time.perf_counter() - t_1203
        t2new -= tmp + tmp.transpose(1, 0, 3, 2)

    else:
        tmp = 2 * lib.einsum("akic,kjcb->ijab", Wvoov, t2)
        tmp -= lib.einsum("akci,kjcb->ijab", Wvovo, t2)

        t2new += (tmp + tmp.transpose(1, 0, 3, 2))

        tmp = lib.einsum("akic,kjbc->ijab", Wvoov, t2)
        t2new -= (tmp + tmp.transpose(1, 0, 3, 2))

        tmp = lib.einsum("bkci,kjac->ijab", Wvovo, t2)
        t2new -= (tmp + tmp.transpose(1, 0, 3, 2))

    # Term 7
    tmp = lib.einsum("akij,kb->ijab", eris_ovoo.transpose(1, 3, 0, 2).conj(), t1) 
    t2new -= (tmp + tmp.transpose(1, 0, 3, 2))

    # Term 8
    tmp = lib.einsum("iabc,jc->ijab", np.asarray(eris_ovvv).conj(), t1)
    t2new += (tmp + tmp.transpose(1, 0, 3, 2))

    #
    #
    eia = mo_e_o[:, None] - mo_e_v
    eijab = lib.direct_sum("ia,jb->ijab", eia, eia)
    t1new /= eia
    t2new /= eijab

    #
    # Timing
    #
    t_update_amps = time.perf_counter() - t_update_amps

    log.debug(f"\nFRI: Compression Time {t_compress:.3f}")
    fri_time = copy.copy(t_compress)
    for k, v in contraction_timings.items():
        log.debug(f"FRI: Contraction {k} Time: {v:.3f} (s)")
        fri_time += v
    fri_time_frac = (fri_time) / t_update_amps
    log.debug(
        f"FRI: CCSD Total Time {t_update_amps:.3f} FRI-related fraction = {fri_time_frac:.3f}\n"
    )

    return t1new, t2new


#
# Intermediates
#


def linear_Foo(t1, t2, eris):
    nocc, nvir = t1.shape
    foo = eris.fock[:nocc, :nocc]
    eris_ovov = np.asarray(eris.ovov)
    Fki = np.zeros_like(foo)
    # Fki = 2 * lib.einsum("kcld,ilcd->ki", eris_ovov, t2)
    # Fki -= lib.einsum("kdlc,ilcd->ki", eris_ovov, t2)
    # Fki = 2 * lib.einsum("kcld,ic,ld->ki", eris_ovov, t1, t1)
    # Fki -= lib.einsum("kdlc,ic,ld->ki", eris_ovov, t1, t1)
    Fki += foo
    return Fki


def linear_Fvv(t1, t2, eris):
    nocc, nvir = t1.shape
    fvv = eris.fock[nocc:, nocc:]
    eris_ovov = np.asarray(eris.ovov)
    Fac = np.zeros_like(fvv)
    # Fac = -2 * lib.einsum("kcld,klad->ac", eris_ovov, t2)
    # Fac += lib.einsum("kdlc,klad->ac", eris_ovov, t2)
    # Fac = -2 * lib.einsum("kcld,ka,ld->ac", eris_ovov, t1, t1)
    # Fac += lib.einsum("kdlc,ka,ld->ac", eris_ovov, t1, t1)
    Fac += fvv
    return Fac


def linear_Fov(t1, t2, eris):
    nocc, nvir = t1.shape
    fov = eris.fock[:nocc, nocc:]
    eris_ovov = np.asarray(eris.ovov)
    Fkc = np.zeros_like(fov)
    # Fkc = 2 * np.einsum("kcld,ld->kc", eris_ovov, t1)
    # Fkc -= np.einsum("kdlc,ld->kc", eris_ovov, t1)
    Fkc += fov
    return Fkc


def linear_Loo(t1, t2, eris):
    nocc, nvir = t1.shape
    fov = eris.fock[:nocc, nocc:]
    # Lki = linear_Foo(t1, t2, eris) + np.einsum("kc,ic->ki", fov, t1)
    Lki = eris.fock[:nocc, :nocc].copy()  # + np.einsum("kc,ic->ki", fov, t1)
    # eris_ovoo = np.asarray(eris.ovoo)
    # Lki += 2 * np.einsum("lcki,lc->ki", eris_ovoo, t1)
    # Lki -= np.einsum("kcli,lc->ki", eris_ovoo, t1)
    return Lki


def linear_Lvv(t1, t2, eris):
    nocc, nvir = t1.shape
    fov = eris.fock[:nocc, nocc:]
    # Lac = linear_Fvv(t1, t2, eris) - np.einsum("kc,ka->ac", fov, t1)
    Lac = eris.fock[nocc:, nocc:].copy()  # - np.einsum("kc,ka->ac", fov, t1)
    # eris_ovvv = np.asarray(eris.get_ovvv())
    # Lac += 2 * np.einsum("kdac,kd->ac", eris_ovvv, t1)
    # Lac -= np.einsum("kcad,kd->ac", eris_ovvv, t1)
    return Lac


### Eqs. (42)-(45) "chi"


def linear_Woooo(t1, t2, eris):
    # eris_ovoo = np.asarray(eris.ovoo)
    # Wklij = lib.einsum("lcki,jc->klij", eris_ovoo, t1)
    # Wklij += lib.einsum("kclj,ic->klij", eris_ovoo, t1)
    # eris_ovov = np.asarray(eris.ovov)
    # Wklij += lib.einsum("kcld,ijcd->klij", eris_ovov, t2)
    # Wklij += lib.einsum("kcld,ic,jd->klij", eris_ovov, t1, t1)
    # Wklij += np.asarray(eris.oooo).transpose(0, 2, 1, 3)
    Wklij = np.asarray(eris.oooo).transpose(0, 2, 1, 3)
    return Wklij


def linear_Wvvvv(t1, t2, eris):
    # Incore
    # eris_ovvv = np.asarray(eris.get_ovvv())
    # Wabcd = lib.einsum("kdac,kb->abcd", eris_ovvv, -t1)
    # Wabcd -= lib.einsum("kcbd,ka->abcd", eris_ovvv, t1)
    # Wabcd += np.asarray(_get_vvvv(eris)).transpose(0, 2, 1, 3)
    Wabcd = np.asarray(imd._get_vvvv(eris)).transpose(0, 2, 1, 3)
    return Wabcd


def linear_Wvoov(t1, t2, eris):
    # eris_ovvv = np.asarray(eris.get_ovvv())
    # eris_ovoo = np.asarray(eris.ovoo)
    # Wakic = lib.einsum("kcad,id->akic", eris_ovvv, t1)
    # Wakic -= lib.einsum("kcli,la->akic", eris_ovoo, t1)
    # Wakic += np.asarray(eris.ovvo).transpose(2, 0, 3, 1)
    # eris_ovov = np.asarray(eris.ovov)
    # Wakic -= 0.5 * lib.einsum("ldkc,ilda->akic", eris_ovov, t2)
    # Wakic -= 0.5 * lib.einsum("lckd,ilad->akic", eris_ovov, t2)
    # Wakic -= lib.einsum("ldkc,id,la->akic", eris_ovov, t1, t1)
    # Wakic += lib.einsum("ldkc,ilad->akic", eris_ovov, t2)
    Wakic = np.asarray(eris.ovvo).transpose(2, 0, 3, 1)
    return Wakic


def linear_Wvovo(t1, t2, eris):
    # eris_ovvv = np.asarray(eris.get_ovvv())
    # eris_ovoo = np.asarray(eris.ovoo)
    # Wakci = lib.einsum("kdac,id->akci", eris_ovvv, t1)
    # Wakci -= lib.einsum("lcki,la->akci", eris_ovoo, t1)
    # Wakci += np.asarray(eris.oovv).transpose(2, 0, 3, 1)
    # eris_ovov = np.asarray(eris.ovov)
    # Wakci -= 0.5 * lib.einsum("lckd,ilda->akci", eris_ovov, t2)
    # Wakci -= lib.einsum("lckd,id,la->akci", eris_ovov, t1, t1)
    Wakci = np.asarray(eris.oovv).transpose(2, 0, 3, 1)

    return Wakci


class FRILCCSD(FRICCSD):
    def __init__(
        self,
        mf,
        frozen=None,
        mo_coeff=None,
        mo_occ=None,
        fri_settings=default_fri_settings,
    ):
        mycc = super().__init__(mf, frozen=None, mo_coeff=None, mo_occ=None)
        self.fri_settings = fri_settings

        log = lib.logger.new_logger(self)

        # Redundent because we use FRICCSD.__init__()
        # Check FRI settings
        for k, v in default_fri_settings.items():
            if k not in self.fri_settings.keys():
                log.debug(f"FRI: Setting {k} to {v}")
                fri_settings[k] = v

        # Check contractions
        for c in self.fri_settings["compressed_contractions"]:
            if c not in ALLOWED_CONTRACTIONS:
                raise ValueError(f"Contraction ({c}) is not supported!")

        return mycc

    def update_amps(self, t1, t2, eris):
        return update_amps(self, t1, t2, eris)
