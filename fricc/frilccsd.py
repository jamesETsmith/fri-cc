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
    Ref: Hirata et al., J. Chem. Phys. 120, 2581 (2004) Eqs.(35)-(36)
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
    foo = fock[:nocc, :nocc].copy()
    fvv = fock[nocc:, nocc:].copy()
    t_update_amps = time.time()

    #
    # Compression
    #
    log.debug(f"M_KEEP = {m_keep} of {t2.size}")
    # log.warn(f"|t2|_1 = {np.linalg.norm(t2.ravel(), ord=1)}")
    # np.save("fricc_t2.npy", t2.ravel())
    # exit(0)

    t_compress = time.time()
    # TODO Fix ravel here it's copying for big arrays
    t2_sparse = SparseTensor4d(
        t2.ravel(),
        t2.shape,
        int(m_keep),
        cc.fri_settings["compression"],
        cc.fri_settings["sampling_method"],
        cc.fri_settings["verbose"],
    )
    t_compress = time.time() - t_compress

    #
    # Updating the Amplitudes
    #

    Foo = linear_Foo(t1, t2, eris)  # Contains no t2
    Fvv = linear_Fvv(t1, t2, eris)  # Contains no t2
    Fov = imd.cc_Fov(t1, t2, eris)  # NO t2

    # # Move energy terms to the other side
    Foo[np.diag_indices(nocc)] -= mo_e_o
    Fvv[np.diag_indices(nvir)] -= mo_e_v

    # T1 equation
    t1new = np.zeros_like(t1)
    # t1new = -2 * np.einsum("kc,ka,ic->ia", fov, t1, t1)
    t1new += np.einsum("ac,ic->ia", Fvv, t1)
    t1new += -np.einsum("ki,ka->ia", Foo, t1)
    t1new += 2 * np.einsum("kc,kica->ia", Fov, t2)
    t1new += -np.einsum("kc,ikca->ia", Fov, t2)
    # t1new += np.einsum("kc,ic,ka->ia", Fov, t1, t1)
    t1new += fov.conj()
    t1new += 2 * np.einsum("kcai,kc->ia", eris.ovvo, t1)
    t1new += -np.einsum("kiac,kc->ia", eris.oovv, t1)

    # TODO sparsify ? (O^2V^3)
    eris_ovvv = np.asarray(eris.get_ovvv())
    t1new += 2 * lib.einsum("kdac,ikcd->ia", eris_ovvv, t2)
    t1new += -lib.einsum("kcad,ikcd->ia", eris_ovvv, t2)
    # t1new += 2 * lib.einsum("kdac,kd,ic->ia", eris_ovvv, t1, t1)
    # t1new += -lib.einsum("kcad,kd,ic->ia", eris_ovvv, t1, t1)

    # TODO sparsify ? (O^3V^2)
    eris_ovoo = np.asarray(eris.ovoo, order="C")
    t1new += -2 * lib.einsum("lcki,klac->ia", eris_ovoo, t2)
    t1new += lib.einsum("kcli,klac->ia", eris_ovoo, t2)

    # t1new += -2 * lib.einsum("lcki,lc,ka->ia", eris_ovoo, t1, t1)
    # t1new += lib.einsum("kcli,lc,ka->ia", eris_ovoo, t1, t1)

    #
    # T2 equation
    #

    # tmp2 = lib.einsum("kibc,ka->abic", eris.oovv, -t1)
    tmp2 = np.asarray(eris_ovvv).conj().transpose(1, 3, 0, 2)
    tmp = lib.einsum("abic,jc->ijab", tmp2, t1)
    t2new = tmp + tmp.transpose(1, 0, 3, 2)
    # tmp2 = lib.einsum("kcai,jc->akij", eris.ovvo, t1)
    tmp2 = eris_ovoo.transpose(1, 3, 0, 2).conj()
    tmp = lib.einsum("akij,kb->ijab", tmp2, t1)
    t2new -= tmp + tmp.transpose(1, 0, 3, 2)
    t2new += np.asarray(eris.ovov).conj().transpose(0, 2, 1, 3)

    # Lxx intermediates only contracted with t2
    Loo = linear_Loo(t1, t2, eris)  # Contains no t2
    Lvv = linear_Lvv(t1, t2, eris)  # Contains no t2
    Loo[np.diag_indices(nocc)] -= mo_e_o
    Lvv[np.diag_indices(nvir)] -= mo_e_v

    # TODO sparsify? O^2V^3
    tmp = lib.einsum("ac,ijcb->ijab", Lvv, t2)
    t2new += tmp + tmp.transpose(1, 0, 3, 2)
    tmp = lib.einsum("ki,kjab->ijab", Loo, t2)
    t2new -= tmp + tmp.transpose(1, 0, 3, 2)

    if "O^4V^2" in compressed_contractions:
        Woooo = sparse_cc_Woooo(t1, t2_sparse_alt, eris, contraction_timings)
    else:
        Woooo = linear_Woooo(t1, t2, eris)

    if "O^3V^3" in compressed_contractions:
        Wvoov = sparse_cc_Wvoov(t1, t2_sparse_alt, eris, contraction_timings)
        Wvovo = sparse_cc_Wvovo(t1, t2_sparse_alt, eris, contraction_timings)

    else:
        Wvoov = linear_Wvoov(t1, t2, eris)
        Wvovo = linear_Wvovo(t1, t2, eris)

    Wvvvv = imd.cc_Wvvvv(t1, t2, eris)  # t2 isn't used in making this intermediate

    # Splitting up some of the taus

    if "O^4V^2" in compressed_contractions:
        # t2new += lib.einsum("klij,ka,lb->ijab", Woooo, t1, t1)
        t_0101 = time.time()
        contract_DTSpT(Woooo, t2_sparse, t2new, "0101")
        contraction_timings["0101"] = time.time() - t_0101
    else:
        tau = t2  # + np.einsum("ia,jb->ijab", t1, t1)
        t2new += lib.einsum("klij,klab->ijab", Woooo, tau)

    # FRI-Compressed contraction
    # t2new += lib.einsum("abcd,ia,jb->ijab", Wvvvv, t1, t1)
    if "O^2V^4" in compressed_contractions:

        t_2323 = time.time()
        contract_DTSpT(Wvvvv, t2_sparse, t2new, "2323")
        contraction_timings["2323"] = time.time() - t_2323
    else:
        t2new += lib.einsum("abcd,ijcd->ijab", Wvvvv, t2)

    # FRI-Compressed contraction
    if "O^3V^3" in compressed_contractions:
        tmp = np.zeros_like(t2new)
        t_1302 = time.time()
        contract_DTSpT(Wvoov, t2_sparse, tmp, "1302")
        contraction_timings["1302"] = time.time() - t_1302

        t_1202 = time.time()
        contract_DTSpT(Wvovo, t2_sparse, tmp, "1202")
        contraction_timings["1202"] = time.time() - t_1202

        t2new += tmp + tmp.transpose(1, 0, 3, 2)

        tmp = np.zeros_like(t2new)
        t_1303 = time.time()
        contract_DTSpT(Wvoov, t2_sparse, tmp, "1303")
        contraction_timings["1303"] = time.time() - t_1303

        t2new -= tmp + tmp.transpose(1, 0, 3, 2)

        tmp = np.zeros_like(t2new)
        t_1203 = time.time()
        contract_DTSpT(Wvovo, t2_sparse, tmp, "1203")
        contraction_timings["1203"] = time.time() - t_1203
        t2new -= tmp + tmp.transpose(1, 0, 3, 2)

    else:
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

    #
    # Timing
    #
    t_update_amps = time.time() - t_update_amps

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
    Fki = 2 * lib.einsum("kcld,ilcd->ki", eris_ovov, t2)
    Fki -= lib.einsum("kdlc,ilcd->ki", eris_ovov, t2)
    # Fki = 2 * lib.einsum("kcld,ic,ld->ki", eris_ovov, t1, t1)
    # Fki -= lib.einsum("kdlc,ic,ld->ki", eris_ovov, t1, t1)
    Fki += foo
    return Fki


def linear_Fvv(t1, t2, eris):
    nocc, nvir = t1.shape
    fvv = eris.fock[nocc:, nocc:]
    eris_ovov = np.asarray(eris.ovov)
    Fac = -2 * lib.einsum("kcld,klad->ac", eris_ovov, t2)
    Fac += lib.einsum("kdlc,klad->ac", eris_ovov, t2)
    # Fac = -2 * lib.einsum("kcld,ka,ld->ac", eris_ovov, t1, t1)
    # Fac += lib.einsum("kdlc,ka,ld->ac", eris_ovov, t1, t1)
    Fac += fvv
    return Fac


def linear_Loo(t1, t2, eris):
    nocc, nvir = t1.shape
    fov = eris.fock[:nocc, nocc:]
    # Lki = linear_Foo(t1, t2, eris) + np.einsum("kc,ic->ki", fov, t1)
    Lki = eris.fock[:nocc, :nocc] + np.einsum("kc,ic->ki", fov, t1)
    eris_ovoo = np.asarray(eris.ovoo)
    Lki += 2 * np.einsum("lcki,lc->ki", eris_ovoo, t1)
    Lki -= np.einsum("kcli,lc->ki", eris_ovoo, t1)
    return Lki


def linear_Lvv(t1, t2, eris):
    nocc, nvir = t1.shape
    fov = eris.fock[:nocc, nocc:]
    # Lac = linear_Fvv(t1, t2, eris) - np.einsum("kc,ka->ac", fov, t1)
    Lac = eris.fock[nocc:, nocc:] - np.einsum("kc,ka->ac", fov, t1)
    eris_ovvv = np.asarray(eris.get_ovvv())
    Lac += 2 * np.einsum("kdac,kd->ac", eris_ovvv, t1)
    Lac -= np.einsum("kcad,kd->ac", eris_ovvv, t1)
    return Lac


### Eqs. (42)-(45) "chi"


def linear_Woooo(t1, t2, eris):
    eris_ovoo = np.asarray(eris.ovoo)
    Wklij = lib.einsum("lcki,jc->klij", eris_ovoo, t1)
    Wklij += lib.einsum("kclj,ic->klij", eris_ovoo, t1)
    eris_ovov = np.asarray(eris.ovov)
    # Wklij += lib.einsum("kcld,ijcd->klij", eris_ovov, t2)
    # Wklij += lib.einsum("kcld,ic,jd->klij", eris_ovov, t1, t1)
    Wklij += np.asarray(eris.oooo).transpose(0, 2, 1, 3)
    return Wklij


def linear_Wvoov(t1, t2, eris):
    eris_ovvv = np.asarray(eris.get_ovvv())
    eris_ovoo = np.asarray(eris.ovoo)
    Wakic = lib.einsum("kcad,id->akic", eris_ovvv, t1)
    Wakic -= lib.einsum("kcli,la->akic", eris_ovoo, t1)
    Wakic += np.asarray(eris.ovvo).transpose(2, 0, 3, 1)
    # eris_ovov = np.asarray(eris.ovov)
    # Wakic -= 0.5 * lib.einsum("ldkc,ilda->akic", eris_ovov, t2)
    # Wakic -= 0.5 * lib.einsum("lckd,ilad->akic", eris_ovov, t2)
    # Wakic -= lib.einsum("ldkc,id,la->akic", eris_ovov, t1, t1)
    # Wakic += lib.einsum("ldkc,ilad->akic", eris_ovov, t2)
    return Wakic


def linear_Wvovo(t1, t2, eris):
    eris_ovvv = np.asarray(eris.get_ovvv())
    eris_ovoo = np.asarray(eris.ovoo)
    Wakci = lib.einsum("kdac,id->akci", eris_ovvv, t1)
    Wakci -= lib.einsum("lcki,la->akci", eris_ovoo, t1)
    Wakci += np.asarray(eris.oovv).transpose(2, 0, 3, 1)
    # eris_ovov = np.asarray(eris.ovov)
    # Wakci -= 0.5 * lib.einsum("lckd,ilda->akci", eris_ovov, t2)
    # Wakci -= lib.einsum("lckd,id,la->akci", eris_ovov, t1, t1)
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
        # # Check FRI settings
        # for k, v in default_fri_settings.items():
        #     if k not in self.fri_settings.keys():
        #         log.debug(f"FRI: Setting {k} to {v}")
        #         fri_settings[k] = v

        # # Check contractions
        # for c in self.fri_settings["compressed_contractions"]:
        #     if c not in ALLOWED_CONTRACTIONS:
        #         raise ValueError(f"Contraction ({c}) is not supported!")

        return mycc

    def update_amps(self, t1, t2, eris):
        return update_amps(self, t1, t2, eris)
