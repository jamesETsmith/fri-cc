import numpy as np
from pyscf import gto, scf, cc, lib
from pyscf.cc import rintermediates as imd
from pyscf.cc import ccsd

from fricc import update_amps_wrapper


def update_amps(cc, t1, t2, eris):

    intermediates = {
        "t1_new": None,
        "t2_new": None,
    }  # data that we'll return as we check things line by line

    # Ref: Hirata et al., J. Chem. Phys. 120, 2581 (2004) Eqs.(35)-(36)
    assert isinstance(eris, ccsd._ChemistsERIs)
    nocc, nvir = t1.shape
    fock = eris.fock
    mo_e_o = eris.mo_energy[:nocc]
    mo_e_v = eris.mo_energy[nocc:] + cc.level_shift

    fov = fock[:nocc, nocc:].copy()
    foo = fock[:nocc, :nocc].copy()
    fvv = fock[nocc:, nocc:].copy()

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

    # # T2 equation
    tmp2 = lib.einsum("kibc,ka->abic", eris.oovv, -t1)
    tmp2 += np.asarray(eris_ovvv).conj().transpose(1, 3, 0, 2)
    tmp = lib.einsum("abic,jc->ijab", tmp2, t1)
    t2new = tmp + tmp.transpose(1, 0, 3, 2)
    tmp2 = lib.einsum("kcai,jc->akij", eris.ovvo, t1)
    tmp2 += eris_ovoo.transpose(1, 3, 0, 2).conj()
    tmp = lib.einsum("akij,kb->ijab", tmp2, t1)
    t2new -= tmp + tmp.transpose(1, 0, 3, 2)
    t2new += np.asarray(eris.ovov).conj().transpose(0, 2, 1, 3)

    # if cc.cc2:
    #     Woooo2 = np.asarray(eris.oooo).transpose(0, 2, 1, 3).copy()
    #     Woooo2 += lib.einsum("lcki,jc->klij", eris_ovoo, t1)
    #     Woooo2 += lib.einsum("kclj,ic->klij", eris_ovoo, t1)
    #     Woooo2 += lib.einsum("kcld,ic,jd->klij", eris.ovov, t1, t1)
    #     t2new += lib.einsum("klij,ka,lb->ijab", Woooo2, t1, t1)
    #     Wvvvv = lib.einsum("kcbd,ka->abcd", eris_ovvv, -t1)
    #     Wvvvv = Wvvvv + Wvvvv.transpose(1, 0, 3, 2)
    #     Wvvvv += np.asarray(eris.vvvv).transpose(0, 2, 1, 3)
    #     t2new += lib.einsum("abcd,ic,jd->ijab", Wvvvv, t1, t1)
    #     Lvv2 = fvv - np.einsum("kc,ka->ac", fov, t1)
    #     Lvv2 -= np.diag(np.diag(fvv))
    #     tmp = lib.einsum("ac,ijcb->ijab", Lvv2, t2)
    #     t2new += tmp + tmp.transpose(1, 0, 3, 2)
    #     Loo2 = foo + np.einsum("kc,ic->ki", fov, t1)
    #     Loo2 -= np.diag(np.diag(foo))
    #     tmp = lib.einsum("ki,kjab->ijab", Loo2, t2)
    #     t2new -= tmp + tmp.transpose(1, 0, 3, 2)
    # else:
    Loo = imd.Loo(t1, t2, eris)
    Lvv = imd.Lvv(t1, t2, eris)
    Loo[np.diag_indices(nocc)] -= mo_e_o
    Lvv[np.diag_indices(nvir)] -= mo_e_v

    Woooo = imd.cc_Woooo(t1, t2, eris)
    Wvoov = imd.cc_Wvoov(t1, t2, eris)
    Wvovo = imd.cc_Wvovo(t1, t2, eris)
    Wvvvv = imd.cc_Wvvvv(t1, t2, eris)

    tau = t2 + np.einsum("ia,jb->ijab", t1, t1)
    t2new += lib.einsum("klij,klab->ijab", Woooo, tau)
    t2new += lib.einsum("abcd,ijcd->ijab", Wvvvv, tau)
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

    # TOOD
    # intermediates["t1_new"] = t1new.copy()
    # intermediates["t2_new"] = t2new.copy()
    # return intermediates

    return t1new, t2new


npt = np.testing
ERROR_TOL = 1e-14


# Generate CC quantities
mol = gto.M(atom="H 0 -1 -1; O 0 0 0; H 0 1.2 -1;", basis="3-21g")
mf = scf.RHF(mol).run()
mycc = cc.CCSD(mf)
mycc.kernel()

# unpack them
t1, t2 = (mycc.t1, mycc.t2)
print(f"t1 shape {t1.shape}")
eris = mycc.ao2mo()
nocc, nvirt = t1.shape


def test_update_amps():
    print()
    print("#" * 80)
    print("Testing `update_amps()`")

    pyscf_t1_new, pyscf_t2_new = update_amps(mycc, t1, t2, eris)
    # pyscf_t1_new = intermediates["t1_new"]
    # pyscf_t2_new = intermediates["t2_new"]

    my_t1_new, my_t2_new = update_amps_wrapper(t1, t2, eris)

    t1_error = np.linalg.norm(pyscf_t1_new - my_t1_new)
    print(f"Error in t1_new {t1_error:.2e}")
    assert t1_error < ERROR_TOL

    t2_error = np.linalg.norm(pyscf_t2_new - my_t2_new)
    print(f"Error in t2_new {t2_error:.2e}")
    assert t2_error < ERROR_TOL