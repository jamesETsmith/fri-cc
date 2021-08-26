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

# Deprecated
from .py_rccsd import update_amps as my_update_amps

numpy = np


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

    m_keep = cc.fri_settings["m_keep"]
    if m_keep > t2.size or m_keep == 0:
        raise ValueError(
            "Bad m_keep value! The following condition wasn't met 0 < m_keep <= t2.size"
        )

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
    log = lib.logger.new_logger(cc)

    #
    # Compression
    #
    t_compress = time.time()
    log.debug(f"M_KEEP = {m_keep} of {t2.size}")
    t2_sparse = SparseTensor4d(
        t2.ravel(),
        t2.shape,
        m_keep,
        cc.fri_settings["compression"],
        cc.fri_settings["sampling_method"],
        cc.fri_settings["verbose"],
    )
    t_compress = time.time() - t_compress

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
    eris_ovvv = np.asarray(eris.get_ovvv())
    t1new += 2 * lib.einsum("kdac,ikcd->ia", eris_ovvv, t2)
    t1new += -lib.einsum("kcad,ikcd->ia", eris_ovvv, t2)
    t1new += 2 * lib.einsum("kdac,kd,ic->ia", eris_ovvv, t1, t1)
    t1new += -lib.einsum("kcad,kd,ic->ia", eris_ovvv, t1, t1)
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

    Woooo = imd.cc_Woooo(t1, t2, eris)
    Wvoov = imd.cc_Wvoov(t1, t2, eris)
    Wvovo = imd.cc_Wvovo(t1, t2, eris)
    Wvvvv = imd.cc_Wvvvv(t1, t2, eris)

    # Splitting up some of the taus
    tau = t2 + np.einsum("ia,jb->ijab", t1, t1)
    t2new += lib.einsum("klij,klab->ijab", Woooo, tau)

    #
    # ORIGINAL: t2new += lib.einsum("abcd,ijcd->ijab", Wvvvv, tau)
    # t2new += lib.einsum("abcd,ijcd->ijab", Wvvvv, t2)
    t2new += lib.einsum("abcd,ia,jb->ijab", Wvvvv, t1, t1)
    t_2323 = time.time()
    contract_DTSpT(Wvvvv.ravel(), t2_sparse, t2new.ravel(), "2323")
    t_2323 = time.time() - t_2323

    tmp = lib.einsum("ac,ijcb->ijab", Lvv, t2)
    t2new += tmp + tmp.transpose(1, 0, 3, 2)
    tmp = lib.einsum("ki,kjab->ijab", Loo, t2)
    t2new -= tmp + tmp.transpose(1, 0, 3, 2)
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
    fri_time_frac = (t_compress + t_2323) / t_update_amps

    log.debug(f"FRI: Compression Time {t_compress:.3f}")
    log.debug(f"FRI: Contraction Time {t_2323:.3f}")
    log.debug(
        f"FRI: CCSD Total Time {t_update_amps:.3f} FRI-related fraction = {fri_time_frac:.3f}"
    )

    return t1new, t2new


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
    log.info("Init E_corr(CCSD) = %.15g", eccsd)

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
        if mycc.iterative_damping < 1.0:
            alpha = mycc.iterative_damping
            t1new = (1 - alpha) * t1 + alpha * t1new
            t2new *= alpha
            t2new += (1 - alpha) * t2
        t1, t2 = t1new, t2new
        t1new = t2new = None
        t1, t2 = mycc.run_diis(t1, t2, istep, normt, eccsd - eold, adiis)
        eold, eccsd = eccsd, mycc.energy(t1, t2, eris)

        # Save old energies
        mycc.energies.append(copy.deepcopy(eccsd))
        # print(mycc.energies)
        log.info(
            "cycle = %d  E_corr(CCSD) = %.15g  dE = %.9g  norm(t1,t2) = %.6g",
            istep + 1,
            eccsd,
            eccsd - eold,
            normt,
        )
        cput1 = log.timer("CCSD iter", *cput1)
        if abs(eccsd - eold) < tol and normt < tolnormt:
            conv = True
            break
    log.timer("CCSD", *cput0)
    return conv, eccsd, t1, t2


#
# Deprecated
#
def update_amps_wrapper(t1, t2, eris):
    nocc, nvirt = t1.shape
    t1_new = np.zeros_like(t1, order="C")
    t2_new = np.zeros_like(t2, order="C")

    fock = eris.fock
    fov = np.ascontiguousarray(fock[:nocc, nocc:])
    foo = np.ascontiguousarray(fock[:nocc, :nocc])
    fvv = np.ascontiguousarray(fock[nocc:, nocc:])

    my_update_amps(
        t1,
        t2.ravel(),
        t1_new,
        t2_new.ravel(),
        foo.copy(),
        fov.copy(),
        fvv.copy(),
        eris.oooo.ravel(),
        eris.ovoo.ravel(),
        eris.oovv.ravel(),
        eris.ovvo.ravel(),
        eris.ovov.ravel(),
        eris.get_ovvv().ravel(),
        imd._get_vvvv(eris).ravel(),
        np.ascontiguousarray(eris.mo_energy),
    )

    return t1_new, t2_new


def mcmc_std(vals: np.ndarray) -> list:
    tau_f = integrated_time(vals)[0]
    # print(tau_f, vals.size, np.var(vals))
    return np.sqrt(tau_f / vals.size * np.var(vals)), tau_f
