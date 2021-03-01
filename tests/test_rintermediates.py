import numpy as np
from pyscf import gto, scf, cc, lib
from pyscf.cc.rintermediates import (
    cc_Foo,
    cc_Fvv,
    cc_Fov,
    Loo,
    Lvv,
    cc_Woooo,
    _get_vvvv,
    cc_Wvvvv,
    cc_Wvoov,
)

from fricc.py_rccsd import (
    make_Foo,
    make_Fvv,
    make_Fov,
    make_Loo,
    make_Lvv,
    make_Woooo,
    make_Wvvvv,
    make_Wvoov,
    make_Wvovo,
)

# Testing settings
npt = np.testing
ERROR_TOL = 1e-14


# Generate CC quantities
mol = gto.M(atom="H 0 -1 -1; O 0 0 0; H 0 1.2 -1;", basis="ccpvdz")
mf = scf.RHF(mol).run()
mycc = cc.CCSD(mf)
mycc.kernel()

# unpack them
t1, t2 = (mycc.t1, mycc.t2)
print(f"t1 shape {t1.shape}")
eris = mycc.ao2mo()
nocc, nvirt = t1.shape


def test_Foo():
    print()
    print("#" * 80)
    print("Foo TEST")

    Foo = cc_Foo(t1, t2, eris)
    print(Foo.shape)

    fock_oo = eris.fock[:nocc, :nocc]

    # My C++ code
    myFoo = np.zeros_like(fock_oo)
    make_Foo(t1, t2.ravel(), np.ascontiguousarray(fock_oo), eris.ovov.ravel(), myFoo)

    # Error
    error = np.linalg.norm(myFoo - Foo)
    print(f"Error in Foo {error:.2e}")
    assert error < ERROR_TOL


def test_Fvv():
    print()
    print("#" * 80)
    print("Fvv TEST")

    Fvv = cc_Fvv(t1, t2, eris)
    fock_vv = eris.fock[nocc:, nocc:]
    myFvv = np.zeros_like(fock_vv)
    make_Fvv(t1, t2.ravel(), np.ascontiguousarray(fock_vv), eris.ovov.ravel(), myFvv)

    error = np.linalg.norm(myFvv - Fvv)
    print(f"Error in Fvv {error:.2e}")
    assert error < ERROR_TOL


def test_Fov():
    print()
    print("#" * 80)
    print("Fov TEST")

    Fov = cc_Fov(t1, t2, eris)
    fock_ov = eris.fock[:nocc, nocc:]
    myFov = np.zeros_like(fock_ov)
    make_Fov(t1, t2.ravel(), np.ascontiguousarray(fock_ov), eris.ovov.ravel(), myFov)

    error = np.linalg.norm(myFov - Fov)
    print(f"Error in Fvv {error:.2e}")
    assert error < ERROR_TOL


def test_Loo():
    print()
    print("#" * 80)
    print("Loo Test")

    myFoo = np.zeros((nocc, nocc))
    make_Foo(
        t1,
        t2.ravel(),
        np.ascontiguousarray(eris.fock[:nocc, :nocc]),
        eris.ovov.ravel(),
        myFoo,
    )

    L_oo = Loo(t1, t2, eris)
    fock_ov = eris.fock[:nocc, nocc:]

    myLoo = np.zeros_like(L_oo)
    make_Loo(
        t1,
        t2.ravel(),
        np.ascontiguousarray(fock_ov),
        eris.ovoo.ravel(),
        myFoo,
        myLoo,
    )
    error = np.linalg.norm(myLoo - L_oo)
    print(f"Error in Fvv {error:.2e}")
    assert error < ERROR_TOL


def test_Lvv():
    print()
    print("#" * 80)
    print("Lvv Test")

    L_vv = Lvv(t1, t2, eris)
    fock_ov = eris.fock[:nocc, nocc:]
    ovvv = np.ascontiguousarray(eris.get_ovvv())
    print(ovvv.shape)

    # C++
    myFvv = np.zeros((nvirt, nvirt))
    make_Fvv(
        t1,
        t2.ravel(),
        np.ascontiguousarray(eris.fock[nocc:, nocc:]),
        eris.ovov.ravel(),
        myFvv,
    )

    myLvv = np.zeros_like(L_vv, order="C")
    make_Lvv(
        t1,
        t2.ravel(),
        np.ascontiguousarray(fock_ov),
        ovvv.ravel(),
        myFvv,
        myLvv,
    )
    error = np.linalg.norm(myLvv - L_vv)
    print(f"Error in Fvv {error:.2e}")
    assert error < ERROR_TOL


def test_Woooo():
    print()
    print("#" * 80)
    print("Woooo Test")
    pyscf_Woooo = cc_Woooo(t1, t2, eris)

    myWoooo = np.zeros_like(pyscf_Woooo, order="C")
    make_Woooo(
        t1,
        t2.ravel(),
        eris.oooo.ravel(),
        eris.ovoo.ravel(),
        eris.ovov.ravel(),
        myWoooo.ravel(),
    )

    error = np.linalg.norm(myWoooo - pyscf_Woooo)
    print(f"Error in Wooo {error:.2e}")
    assert error < ERROR_TOL


def test_Wvvvv():
    print()
    print("#" * 80)
    print("Wvvvv Test")

    pyscf_Wvvvv = cc_Wvvvv(t1, t2, eris)

    ovvv = np.ascontiguousarray(eris.get_ovvv())
    myWvvvv = np.zeros_like(pyscf_Wvvvv, order="C")
    make_Wvvvv(
        t1,
        t2.ravel(),
        ovvv.ravel(),
        _get_vvvv(eris).ravel(),
        myWvvvv.ravel(),
    )

    error = np.linalg.norm(myWvvvv - pyscf_Wvvvv)
    print(f"Error in Wooo {error:.2e}")
    assert error < ERROR_TOL


def test_Wvoov():
    print()
    print("#" * 80)
    print("Wvoov Test")

    pyscf_Wvoov = cc_Wvoov(t1, t2, eris)

    ovoo = np.ascontiguousarray(eris.ovoo)
    ovov = np.ascontiguousarray(eris.ovov)
    ovvo = np.ascontiguousarray(eris.ovvo)
    ovvv = np.ascontiguousarray(eris.get_ovvv())

    myWvoov = np.zeros_like(pyscf_Wvoov, order="C")
    make_Wvoov(
        t1,
        t2.ravel(),
        ovoo.ravel(),
        ovov.ravel(),
        ovvo.ravel(),
        ovvv.ravel(),
        myWvoov.ravel(),
    )

    error = np.linalg.norm(myWvoov - pyscf_Wvoov)
    print(f"Error in Wooo {error:.2e}")
    assert error < ERROR_TOL


def cc_Wvovo(t1, t2, eris):
    eris_ovvv = np.asarray(eris.get_ovvv())
    eris_ovoo = np.asarray(eris.ovoo)
    Wakci = lib.einsum("kdac,id->akci", eris_ovvv, t1)
    Wakci -= lib.einsum("lcki,la->akci", eris_ovoo, t1)
    Wakci += np.asarray(eris.oovv).transpose(2, 0, 3, 1)
    eris_ovov = np.asarray(eris.ovov)
    Wakci -= 0.5 * lib.einsum("lckd,ilda->akci", eris_ovov, t2)
    Wakci -= lib.einsum("lckd,id,la->akci", eris_ovov, t1, t1)
    return Wakci


def test_Wvovo():
    print()
    print("#" * 80)
    print("Wvovo Test")

    pyscf_Wvovo = cc_Wvovo(t1, t2, eris)

    ovoo = np.ascontiguousarray(eris.ovoo)
    ovov = np.ascontiguousarray(eris.ovov)
    oovv = np.ascontiguousarray(eris.oovv)
    ovvv = np.ascontiguousarray(eris.get_ovvv())

    myWvovo = np.zeros_like(pyscf_Wvovo, order="C")
    make_Wvovo(
        t1,
        t2.ravel(),
        ovoo.ravel(),
        ovov.ravel(),
        oovv.ravel(),
        ovvv.ravel(),
        myWvovo.ravel(),
    )

    error = np.linalg.norm(myWvovo - pyscf_Wvovo)
    print(f"Error in Wooo {error:.2e}")
    assert error < ERROR_TOL