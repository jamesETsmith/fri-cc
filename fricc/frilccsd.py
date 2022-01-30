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
from .friccsd import FRICCSD, ALLOWED_CONTRACTIONS

numpy = np
default_fri_settings = {
        "m_keep": 1000,
        "compression": "fri",
        "sampling_method": "systematic",
        "verbose": False,
        "compressed_contractions": ["O^2V^4"],
        "LCCD": False,
        "Independent Compressions": False,
}


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
    Fov = fock[:nocc, nocc:].copy()

    # Move energy terms to the other side
    Foo[np.diag_indices(nocc)] -= mo_e_o
    Fvv[np.diag_indices(nvir)] -= mo_e_v

    ###############
    # T1 equation #
    ###############
    eris_ovoo = np.asarray(eris.ovoo, order="C")
    eris_ovvv = np.asarray(eris.get_ovvv())

    if cc.fri_settings["LCCD"]:
        t1 = np.zeros_like(t1)
        t1new = np.zeros_like(t1)  # DO LCCD
    else:
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
        t1new += 2 * lib.einsum("kdac,ikcd->ia", eris_ovvv, t2)
        t1new += -lib.einsum("kcad,ikcd->ia", eris_ovvv, t2)

        # Term 6
        t1new += -2 * lib.einsum("lcki,klac->ia", eris_ovoo, t2)
        t1new += lib.einsum("kcli,klac->ia", eris_ovoo, t2)

        # Term 7
        t1new += 2 * np.einsum("kc,kica->ia", Fov, t2)
        t1new += -np.einsum("kc,ikca->ia", Fov, t2)

    ###############
    # T2 equation #
    ###############

    # Term 1
    t2new = np.zeros_like(t2)
    t2new += np.asarray(eris.ovov).conj().transpose(0, 2, 1, 3)

    # Term 2
    tmp = lib.einsum("ki,kjab->ijab", Foo, t2)
    t2new -= tmp + tmp.transpose(1, 0, 3, 2)

    # Term 3
    tmp = lib.einsum("ac,ijcb->ijab", Fvv, t2)
    t2new += tmp + tmp.transpose(1, 0, 3, 2)

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
        t2new += lib.einsum("klij,klab->ijab", Woooo, t2)

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

        t2new += tmp + tmp.transpose(1, 0, 3, 2)

        tmp = lib.einsum("akic,kjbc->ijab", Wvoov, t2)
        t2new -= tmp + tmp.transpose(1, 0, 3, 2)

        tmp = lib.einsum("bkci,kjac->ijab", Wvovo, t2)
        t2new -= tmp + tmp.transpose(1, 0, 3, 2)

    # Term 7
    tmp = lib.einsum("akij,kb->ijab", eris_ovoo.transpose(1, 3, 0, 2).conj(), t1)
    t2new -= tmp + tmp.transpose(1, 0, 3, 2)

    # Term 8
    tmp = lib.einsum("iabc,jc->ijab", np.asarray(eris_ovvv).conj(), t1)
    t2new += tmp + tmp.transpose(1, 0, 3, 2)

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


def update_amps_independent_compressions(
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
    t2_sparse = [
        SparseTensor4d(
            t2,
            t2.shape,
            int(m_keep),
            cc.fri_settings["compression"],
            cc.fri_settings["sampling_method"],
            cc.fri_settings["verbose"],
        )
        for _ in range(6)
    ]
    t_compress = time.perf_counter() - t_compress

    #
    # Updating the Amplitudes
    #

    Foo = fock[:nocc, :nocc].copy()
    Fvv = fock[nocc:, nocc:].copy()
    Fov = fock[:nocc, nocc:].copy()

    # Move energy terms to the other side
    Foo[np.diag_indices(nocc)] -= mo_e_o
    Fvv[np.diag_indices(nvir)] -= mo_e_v

    ###############
    # T1 equation #
    ###############
    eris_ovoo = np.asarray(eris.ovoo, order="C")
    eris_ovvv = np.asarray(eris.get_ovvv())

    if cc.fri_settings["LCCD"]:
        t1 = np.zeros_like(t1)
        t1new = np.zeros_like(t1)  # DO LCCD
    else:
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
        t1new += 2 * lib.einsum("kdac,ikcd->ia", eris_ovvv, t2)
        t1new += -lib.einsum("kcad,ikcd->ia", eris_ovvv, t2)

        # Term 6
        t1new += -2 * lib.einsum("lcki,klac->ia", eris_ovoo, t2)
        t1new += lib.einsum("kcli,klac->ia", eris_ovoo, t2)

        # Term 7
        t1new += 2 * np.einsum("kc,kica->ia", Fov, t2)
        t1new += -np.einsum("kc,ikca->ia", Fov, t2)

    ###############
    # T2 equation #
    ###############

    # Term 1
    t2new = np.zeros_like(t2)
    t2new += np.asarray(eris.ovov).conj().transpose(0, 2, 1, 3)

    # Term 2
    tmp = lib.einsum("ki,kjab->ijab", Foo, t2)
    t2new -= tmp + tmp.transpose(1, 0, 3, 2)

    # Term 3
    tmp = lib.einsum("ac,ijcb->ijab", Fvv, t2)
    t2new += tmp + tmp.transpose(1, 0, 3, 2)

    Woooo = np.asarray(eris.oooo).copy().transpose(0, 2, 1, 3)
    Wvoov = np.asarray(eris.ovvo).copy().transpose(2, 0, 3, 1)
    Wvovo = np.asarray(eris.oovv).copy().transpose(2, 0, 3, 1)
    Wvvvv = np.asarray(imd._get_vvvv(eris)).transpose(0, 2, 1, 3)

    # Term 4
    if "O^2V^4" in compressed_contractions:
        t_2323 = time.perf_counter()
        contract_DTSpT(Wvvvv, t2_sparse[0], t2new, "2323")
        contraction_timings["2323"] = time.perf_counter() - t_2323
    else:
        t2new += lib.einsum("abcd,ijcd->ijab", Wvvvv, t2)

    # Term 5
    if "O^4V^2" in compressed_contractions:
        t_0101 = time.perf_counter()
        contract_DTSpT(Woooo, t2_sparse[1], t2new, "0101")
        contraction_timings["0101"] = time.perf_counter() - t_0101
    else:
        t2new += lib.einsum("klij,klab->ijab", Woooo, t2)

    # FRI-Compressed contraction
    # Term 6
    if "O^3V^3" in compressed_contractions:
        tmp = np.zeros_like(t2new)
        t_1302 = time.perf_counter()
        contract_DTSpT(Wvoov, t2_sparse[2], tmp, "1302")
        contraction_timings["1302"] = time.perf_counter() - t_1302

        t_1202 = time.perf_counter()
        contract_DTSpT(Wvovo, t2_sparse[3], tmp, "1202")
        contraction_timings["1202"] = time.perf_counter() - t_1202

        t2new += tmp + tmp.transpose(1, 0, 3, 2)

        tmp = np.zeros_like(t2new)
        t_1303 = time.perf_counter()
        contract_DTSpT(Wvoov, t2_sparse[4], tmp, "1303")
        contraction_timings["1303"] = time.perf_counter() - t_1303

        t2new -= tmp + tmp.transpose(1, 0, 3, 2)

        tmp = np.zeros_like(t2new)
        t_1203 = time.perf_counter()
        contract_DTSpT(Wvovo, t2_sparse[5], tmp, "1203")
        contraction_timings["1203"] = time.perf_counter() - t_1203
        t2new -= tmp + tmp.transpose(1, 0, 3, 2)

    else:
        tmp = 2 * lib.einsum("akic,kjcb->ijab", Wvoov, t2)
        tmp -= lib.einsum("akci,kjcb->ijab", Wvovo, t2)

        t2new += tmp + tmp.transpose(1, 0, 3, 2)

        tmp = lib.einsum("akic,kjbc->ijab", Wvoov, t2)
        t2new -= tmp + tmp.transpose(1, 0, 3, 2)

        tmp = lib.einsum("bkci,kjac->ijab", Wvovo, t2)
        t2new -= tmp + tmp.transpose(1, 0, 3, 2)

    # Term 7
    tmp = lib.einsum("akij,kb->ijab", eris_ovoo.transpose(1, 3, 0, 2).conj(), t1)
    t2new -= tmp + tmp.transpose(1, 0, 3, 2)

    # Term 8
    tmp = lib.einsum("iabc,jc->ijab", np.asarray(eris_ovvv).conj(), t1)
    t2new += tmp + tmp.transpose(1, 0, 3, 2)

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


def lin_cc_energy(mycc, t1=None, t2=None, eris=None):
    """Linearized CC correlation energy"""
    if t1 is None:
        t1 = mycc.t1
    if t2 is None:
        t2 = mycc.t2
    if eris is None:
        eris = mycc.ao2mo()

    nocc, nvir = t1.shape
    fock = eris.fock
    e = 2 * np.einsum("ia,ia", fock[:nocc, nocc:], t1)

    eris_ovov = np.asarray(eris.ovov)
    e += 2 * np.einsum("ijab,iajb", t2, eris_ovov)
    e += -np.einsum("ijab,ibja", t2, eris_ovov)
    if abs(e.imag) > 1e-4:
        logger.warn(mycc, "Non-zero imaginary part found in LinCCSD energy %s", e)
    return e.real


class FRILCCSD(FRICCSD):

    def __init__(
        self,
        mf,
        frozen=None,
        mo_coeff=None,
        mo_occ=None,
        fri_settings=default_fri_settings,
    ):
        mycc = super().__init__(mf, frozen=frozen, mo_coeff=mo_coeff, mo_occ=mo_occ)
        self.fri_settings = fri_settings

        log = lib.logger.new_logger(self)

        # Redundent because we use FRICCSD.__init__()
        # Check FRI settings
        for k, v in default_fri_settings.items():
            if k not in self.fri_settings.keys():
                log.debug(f"FRI: Setting {k} to {v}")
                fri_settings[k] = v
            else:
                log.debug(f"FRI: {k} is set to {v}")

        # Check contractions
        for c in self.fri_settings["compressed_contractions"]:
            if c not in ALLOWED_CONTRACTIONS:
                raise ValueError(f"Contraction ({c}) is not supported!")

        return mycc

    def update_amps(self, t1, t2, eris):
        if self.fri_settings["Independent Compressions"]:
            return update_amps_independent_compressions(self, t1, t2, eris)
        else:
            return update_amps(self, t1, t2, eris)

    def energy(self, t1, t2, eris):
        return lin_cc_energy(self, t1, t2, eris)
