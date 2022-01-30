import time
import numpy as np
import copy

from pyscf.cc import ccsd
from pyscf import lib
from pyscf.lib import logger
from pyscf.cc import rintermediates as imd

from .py_rccsd import SparseTensor4d
from .py_rccsd import contract_DTSpT

numpy = np

default_fri_settings = {
        "m_keep": 1000,
        "compression": "fri",
        "sampling_method": "systematic",
        "verbose": False,
        "compressed_contractions": ["O^2V^4"],
    }


ALLOWED_CONTRACTIONS = ["O^2V^4", "O^3V^3", "O^4V^2"]


def update_amps(
    cc: ccsd.CCSD,
    t1: np.ndarray,
    t2: np.ndarray,
    eris: ccsd._ChemistsERIs,
):
    """Update the CC t1 and t2 amplitudes using a compressed t2 tensor for
    several of the contractions.

    Parameters
    ----------
    cc : ccsd.CCSD
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
    t_update_amps = time.perf_counter()

    #
    # Compression
    #
    log.debug(f"M_KEEP = {m_keep} of {t2.size}")
    # log.warn(f"|t2|_1 = {np.linalg.norm(t2.ravel(), ord=1)}")
    # np.save("fricc_t2.npy", t2.ravel())
    # exit(0)
    t2 = np.ascontiguousarray(t2)

    t_compress = time.perf_counter()
    t2_sparse = SparseTensor4d(
        t2,
        t2.shape,
        int(m_keep),
        cc.fri_settings["compression"],
        cc.fri_settings["sampling_method"],
        cc.fri_settings["verbose"],
    )
    if "O^3V^3" in compressed_contractions or "O^4V^2" in compressed_contractions:
        t2_sparse_alt = SparseTensor4d(
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

    Foo = imd.cc_Foo(t1, t2, eris)
    Fvv = imd.cc_Fvv(t1, t2, eris)
    Fov = imd.cc_Fov(t1, t2, eris)

    # Move energy terms to the other side
    Foo[np.diag_indices(nocc)] -= mo_e_o
    Fvv[np.diag_indices(nvir)] -= mo_e_v

    # T1 equation
    t1new = -2 * np.einsum("kc,ka,ic->ia", fov, t1, t1)
    t1new += np.einsum("ac,ic->ia", Fvv, t1)
    t1new += -np.einsum("ki,ka->ia", Foo, t1)
    t1new += 2 * np.einsum("kc,kica->ia", Fov, t2)
    t1new += -np.einsum("kc,ikca->ia", Fov, t2)
    t1new += np.einsum("kc,ic,ka->ia", Fov, t1, t1)
    t1new += fov.conj()
    t1new += 2 * np.einsum("kcai,kc->ia", eris.ovvo, t1)
    t1new += -np.einsum("kiac,kc->ia", eris.oovv, t1)

    # (O^2V^3)
    eris_ovvv = np.asarray(eris.get_ovvv())
    t1new += 2 * lib.einsum("kdac,ikcd->ia", eris_ovvv, t2)
    t1new += -lib.einsum("kcad,ikcd->ia", eris_ovvv, t2)
    t1new += 2 * lib.einsum("kdac,kd,ic->ia", eris_ovvv, t1, t1)
    t1new += -lib.einsum("kcad,kd,ic->ia", eris_ovvv, t1, t1)

    # (O^3V^2)
    eris_ovoo = np.asarray(eris.ovoo, order="C")
    t1new += -2 * lib.einsum("lcki,klac->ia", eris_ovoo, t2)
    t1new += lib.einsum("kcli,klac->ia", eris_ovoo, t2)

    t1new += -2 * lib.einsum("lcki,lc,ka->ia", eris_ovoo, t1, t1)
    t1new += lib.einsum("kcli,lc,ka->ia", eris_ovoo, t1, t1)

    # T2 equation
    tmp2 = lib.einsum("kibc,ka->abic", eris.oovv, -t1)
    tmp2 += np.asarray(eris_ovvv).conj().transpose(1, 3, 0, 2)
    tmp = lib.einsum("abic,jc->ijab", tmp2, t1)
    t2new = tmp + tmp.transpose(1, 0, 3, 2)
    tmp2 = lib.einsum("kcai,jc->akij", eris.ovvo, t1)
    tmp2 += eris_ovoo.transpose(1, 3, 0, 2).conj()
    tmp = lib.einsum("akij,kb->ijab", tmp2, t1)
    t2new -= tmp + tmp.transpose(1, 0, 3, 2)
    t2new += np.asarray(eris.ovov).conj().transpose(0, 2, 1, 3)

    Loo = imd.Loo(t1, t2, eris)
    Lvv = imd.Lvv(t1, t2, eris)
    Loo[np.diag_indices(nocc)] -= mo_e_o
    Lvv[np.diag_indices(nvir)] -= mo_e_v

    if "O^4V^2" in compressed_contractions:
        Woooo = sparse_cc_Woooo(t1, t2_sparse_alt, eris, contraction_timings)
    else:
        Woooo = imd.cc_Woooo(t1, t2, eris)

    if "O^3V^3" in compressed_contractions:
        Wvoov = sparse_cc_Wvoov(t1, t2_sparse_alt, eris, contraction_timings)
        Wvovo = sparse_cc_Wvovo(t1, t2_sparse_alt, eris, contraction_timings)

    else:
        Wvoov = imd.cc_Wvoov(t1, t2, eris)
        Wvovo = imd.cc_Wvovo(t1, t2, eris)

    Wvvvv = cc_Wvvvv(t1, t2, eris)  # t2 isn't used in making this intermediate

    # Splitting up some of the taus
    if "O^4V^2" in compressed_contractions:
        t2new += lib.einsum("klij,ka,lb->ijab", Woooo, t1, t1)

        t_0101 = time.perf_counter()
        contract_DTSpT(Woooo, t2_sparse, t2new, "0101")
        contraction_timings["0101"] = time.perf_counter() - t_0101

    else:
        tau = t2 + np.einsum("ia,jb->ijab", t1, t1)
        t2new += lib.einsum("klij,klab->ijab", Woooo, tau)

    # FRI-Compressed contraction
    t2new += lib.einsum("abcd,ia,jb->ijab", Wvvvv, t1, t1)
    if "O^2V^4" in compressed_contractions:
        # log.warn(f"|t2_new|_2323 = {np.linalg.norm(t2new.ravel(), ord=1)}")

        t_2323 = time.perf_counter()
        contract_DTSpT(Wvvvv, t2_sparse, t2new, "2323")
        contraction_timings["2323"] = time.perf_counter() - t_2323

        # log.warn(f"|t2_new|_2323 = {np.linalg.norm(t2new.ravel(), ord=1)}")
        # t2new = t2new.reshape(nocc, nocc, nvir, nvir)
    else:
        # log.warn(f"|t2_new|_2323 = {np.linalg.norm(t2new.ravel(), ord=1)}")
        t2new += lib.einsum("abcd,ijcd->ijab", Wvvvv, t2)
        # log.warn(f"|t2_new|_2323 = {np.linalg.norm(t2new.ravel(), ord=1)}")

    # TODO sparsify? O^2V^3
    tmp = lib.einsum("ac,ijcb->ijab", Lvv, t2)
    t2new += tmp + tmp.transpose(1, 0, 3, 2)
    tmp = lib.einsum("ki,kjab->ijab", Loo, t2)
    t2new -= tmp + tmp.transpose(1, 0, 3, 2)

    # FRI-Compressed contraction
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
# Constructing intermediates with sparse tensor contractions
#


def sparse_cc_Woooo(t1, t2_sparse, eris, contraction_timings):
    eris_ovoo = np.asarray(eris.ovoo)
    Wklij = lib.einsum("lcki,jc->klij", eris_ovoo, t1)
    Wklij += lib.einsum("kclj,ic->klij", eris_ovoo, t1)

    eris_ovov = np.asarray(eris.ovov)
    t_1323 = time.perf_counter()
    contract_DTSpT(eris_ovov, t2_sparse, Wklij, "1323")
    contraction_timings["1323"] = time.perf_counter() - t_1323
    # Wklij += lib.einsum("kcld,ijcd->klij", eris_ovov, t2)

    Wklij += lib.einsum("kcld,ic,jd->klij", eris_ovov, t1, t1)
    Wklij += np.asarray(eris.oooo).transpose(0, 2, 1, 3)
    return Wklij


def cc_Wvvvv(t1, t2, eris):
    # Incore
    eris_ovvv = np.asarray(eris.get_ovvv())
    Wabcd = lib.einsum("kdac,kb->abcd", eris_ovvv, -t1)
    Wabcd -= lib.einsum("kcbd,ka->abcd", eris_ovvv, t1)
    Wabcd += np.asarray(imd._get_vvvv(eris)).transpose(0, 2, 1, 3)
    return Wabcd


def sparse_cc_Wvoov(t1, t2_sparse, eris, contraction_timings):
    eris_ovvv = np.asarray(eris.get_ovvv())
    eris_ovoo = np.asarray(eris.ovoo)
    Wakic = lib.einsum("kcad,id->akic", eris_ovvv, t1, order="C")
    Wakic -= lib.einsum("kcli,la->akic", eris_ovoo, t1)
    Wakic += np.asarray(eris.ovvo).transpose(2, 0, 3, 1)

    eris_ovov = np.asarray(eris.ovov)

    # Wakic -= 0.5 * lib.einsum("ldkc,ilda->akic", eris_ovov, t2)
    # Wakic -= 0.5 * lib.einsum("lckd,ilad->akic", eris_ovov, t2)
    # Wakic += lib.einsum("ldkc,ilad->akic", eris_ovov, t2)

    t_0112 = time.perf_counter()
    contract_DTSpT(eris_ovov, t2_sparse, Wakic, "0112")
    contraction_timings["0112"] = time.perf_counter() - t_0112

    t_0313 = time.perf_counter()
    contract_DTSpT(eris_ovov, t2_sparse, Wakic, "0313")
    contraction_timings["0313"] = time.perf_counter() - t_0313

    t_0113 = time.perf_counter()
    contract_DTSpT(eris_ovov, t2_sparse, Wakic, "0113")
    contraction_timings["0113"] = time.perf_counter() - t_0113

    Wakic -= lib.einsum("ldkc,id,la->akic", eris_ovov, t1, t1)
    return Wakic


def sparse_cc_Wvovo(t1, t2_sparse, eris, contraction_timings):
    eris_ovvv = np.asarray(eris.get_ovvv())
    eris_ovoo = np.asarray(eris.ovoo)
    Wakci = lib.einsum("kdac,id->akci", eris_ovvv, t1)
    Wakci -= lib.einsum("lcki,la->akci", eris_ovoo, t1)
    Wakci += np.asarray(eris.oovv).transpose(2, 0, 3, 1)

    eris_ovov = np.asarray(eris.ovov)
    t_0312 = time.perf_counter()
    contract_DTSpT(eris_ovov, t2_sparse, Wakci, "0312")
    contraction_timings["0312"] = time.perf_counter() - t_0312

    Wakci -= lib.einsum("lckd,id,la->akci", eris_ovov, t1, t1)
    return Wakci


def kernel(
    mycc,
    eris=None,
    t1=None,
    t2=None,
    max_cycle=50,
    tol=1e-8,
    tolnormt=1e-6,
    verbose=None,
):
    # print("MAX_CYCLE=", mycc.max_cycle, max_cycle)
    max_cycle = mycc.max_cycle  # Hack to force pyscf to respect max_cycles

    log = logger.new_logger(mycc, verbose)
    if eris is None:
        eris = mycc.ao2mo(mycc.mo_coeff)
    if t1 is None and t2 is None:
        t1, t2 = mycc.get_init_guess(eris)
    elif t2 is None:
        t2 = mycc.get_init_guess(eris)[1]

    cput1 = cput0 = (logger.process_clock(), logger.perf_counter())
    eold = 0
    eccsd = mycc.energy(t1, t2, eris)
    log.info("Init E_corr(%s) = %.15g", mycc.__class__.__name__, eccsd)

    if isinstance(mycc.diis, lib.diis.DIIS):
        adiis = mycc.diis
    elif mycc.diis:
        adiis = lib.diis.DIIS(mycc, mycc.diis_file, incore=mycc.incore_complete)
        adiis.space = mycc.diis_space
    else:
        adiis = None

    # Create list of energies so we can access them easily later
    mycc.energies = []
    conv = False
    for istep in range(max_cycle):
        t1new, t2new = mycc.update_amps(t1, t2, eris)
        tmpvec = mycc.amplitudes_to_vector(t1new, t2new)
        tmpvec -= mycc.amplitudes_to_vector(t1, t2)
        normt = numpy.linalg.norm(tmpvec)
        tmpvec = None

        t1, t2 = t1new, t2new
        t1new = t2new = None
        # Skip DIIS
        # t1, t2 = mycc.run_diis(t1, t2, istep, normt, eccsd - eold, adiis)
        eold, eccsd = eccsd, mycc.energy(t1, t2, eris)

        # Save old energies
        mycc.energies.append(copy.deepcopy(eccsd))
        # print(mycc.energies)
        log.info(
            "cycle = %d  E_corr(%s) = %.15g  dE = %.9g  norm(t1,t2) = %.6g",
            istep + 1,
            mycc.__class__.__name__,
            eccsd,
            eccsd - eold,
            normt,
        )
        cput1 = log.timer(f"{mycc.__class__.__name__} iter", *cput1)
        if abs(eccsd - eold) < tol and normt < tolnormt:
            conv = True
            break
        if np.isnan(eccsd - eold):
            conv = False
            log.warn(f"{mycc.__class__.__name__} FAILED THERE ARE NANS")
            break
        elif abs(eccsd - eold) > 1e5:
            conv = False
            log.warn(
                f"{mycc.__class__.__name__} FAILED BECAUSE dE = {abs(eccsd - eold)}"
            )
            break

    log.timer(f"{mycc.__class__.__name__}", *cput0)
    return conv, eccsd, t1, t2


#
# FRI-CCSD class
#
class FRICCSD(ccsd.CCSD):

    def __init__(
        self,
        mf,
        fri_settings=default_fri_settings,
        frozen=None,
        mo_coeff=None,
        mo_occ=None,
    ):
        mycc = super().__init__(mf, frozen=frozen, mo_coeff=mo_coeff, mo_occ=mo_occ)
        self._fri_settings = fri_settings

        log = lib.logger.new_logger(self)

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

    @property
    def fri_settings(self):
        return self._fri_settings

    def ccsd(self, t1=None, t2=None, eris=None):
        assert self.mo_coeff is not None
        assert self.mo_occ is not None

        if self.verbose >= logger.WARN:
            self.check_sanity()
        self.dump_flags()

        if eris is None:
            eris = self.ao2mo(self.mo_coeff)

        self.e_hf = getattr(eris, "e_hf", None)
        if self.e_hf is None:
            self.e_hf = self._scf.e_tot

        self.converged, self.e_corr, self.t1, self.t2 = kernel(
            self,
            eris,
            t1,
            t2,
            max_cycle=self.max_cycle,
            tol=self.conv_tol,
            tolnormt=self.conv_tol_normt,
            verbose=self.verbose,
        )
        self._finalize()
        return self.e_corr, self.t1, self.t2

    def update_amps(self, t1, t2, eris):
        return update_amps(self, t1, t2, eris)
