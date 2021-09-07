import numpy as np
import pytest
from fricc import SparseTensor4d, contract_DTSpT

npt = np.testing
np.random.seed(20)


@pytest.mark.parametrize(
    "no,nv,frac",
    [
        (4, 8, 0.1),
        (10, 15, 0.001),
        (100, 2, 0.001),
    ],
)
def test_sparse_init(no, nv, frac):
    print()

    # Setup random array
    a = np.ascontiguousarray(np.random.rand(no, no, nv, nv))
    m = int(a.size * frac)

    # Keep the tests small
    if m > 200:
        print(f"M={m}")
        raise ValueError("M >= 1000, choose a smaller matrix or a smaller fraction.")

    # Compress by getting the largest m elements
    a_compressed = SparseTensor4d(a.ravel(), a.shape, m, "largest")
    # a_compressed.print()

    # Check that all elements in a_compressed have the right value/index
    for mi in range(m):
        idx, value = a_compressed.get_element(mi)
        npt.assert_equal(value, a[tuple(idx)])


@pytest.mark.parametrize("no,nv,frac", [(4, 8, 0.1), (5, 10, 0.01), (10, 20, 0.001)])
def test_0101_contraction(no, nv, frac):
    t2 = np.ascontiguousarray(np.random.rand(no, no, nv, nv))
    w = np.ascontiguousarray(np.random.rand(no, no, no, no))

    m = int(t2.size * frac)

    # Keep the tests small
    if m > 200:
        print(f"M={m}")
        raise ValueError("M >= 1000, choose a smaller matrix or a smaller fraction.")

    # Compress by getting the largest m elements
    t2_compressed = SparseTensor4d(t2.ravel(), t2.shape, m, "largest")

    # Zero out the elements of the original t2 array
    idx = np.unravel_index(np.argsort(t2, axis=None), t2.shape)
    for i in range(t2.size - m):
        t2[idx[0][i], idx[1][i], idx[2][i], idx[3][i]] = 0

    t2_new_np = np.einsum("klij,klab->ijab", w, t2, order="C")
    t2_new_fri = np.zeros(t2.shape, order="C")
    contract_DTSpT(w.ravel(), t2_compressed, t2_new_fri.ravel(), "0101")

    npt.assert_almost_equal(t2_new_fri, t2_new_np)


@pytest.mark.parametrize(
    "no,nv,frac",
    [
        (4, 8, 0.1),
        (5, 10, 0.01),
        (10, 20, 0.001),
    ],
)
def test_2323_contraction(no, nv, frac):
    t2 = np.ascontiguousarray(np.random.rand(no, no, nv, nv))
    w = np.ascontiguousarray(np.random.rand(nv, nv, nv, nv))

    m = int(t2.size * frac)

    # Keep the tests small
    if m > 200:
        print(f"M={m}")
        raise ValueError("M >= 1000, choose a smaller matrix or a smaller fraction.")

    # Compress by getting the largest m elements
    t2_compressed = SparseTensor4d(t2.ravel(), t2.shape, m, "largest")

    # Zero out the elements of the original t2 array
    idx = np.unravel_index(np.argsort(t2, axis=None), t2.shape)
    for i in range(t2.size - m):
        t2[idx[0][i], idx[1][i], idx[2][i], idx[3][i]] = 0

    # t2new += lib.einsum('abcd,ijcd->ijab', Wvvvv, tau)
    t2_new_np = np.einsum("abcd,ijcd->ijab", w, t2, order="C")
    t2_new_fri = np.zeros(t2.shape, order="C")
    contract_DTSpT(w.ravel(), t2_compressed, t2_new_fri.ravel(), "2323")
    npt.assert_almost_equal(t2_new_fri, t2_new_np)


@pytest.mark.parametrize("no,nv,frac", [(4, 8, 0.1), (5, 10, 0.01), (10, 20, 0.001)])
def test_1302_contraction(no, nv, frac):
    t2 = np.ascontiguousarray(np.random.rand(no, no, nv, nv))
    w = np.ascontiguousarray(np.random.rand(nv, no, no, nv))

    m = int(t2.size * frac)

    # Keep the tests small
    if m > 200:
        print(f"M={m}")
        raise ValueError("M >= 1000, choose a smaller matrix or a smaller fraction.")

    # Compress by getting the largest m elements
    t2_compressed = SparseTensor4d(t2.ravel(), t2.shape, m, "largest")

    # Zero out the elements of the original t2 array
    idx = np.unravel_index(np.argsort(t2, axis=None), t2.shape)
    for i in range(t2.size - m):
        t2[idx[0][i], idx[1][i], idx[2][i], idx[3][i]] = 0

    t2_new_np = 2 * np.einsum("akic,kjcb->ijab", w, t2, order="C")
    t2_new_fri = np.zeros(t2.shape, order="C")
    contract_DTSpT(w.ravel(), t2_compressed, t2_new_fri.ravel(), "1302")

    npt.assert_almost_equal(t2_new_fri, t2_new_np)


@pytest.mark.parametrize("no,nv,frac", [(4, 8, 0.1), (5, 10, 0.01), (10, 20, 0.001)])
def test_1202_contraction(no, nv, frac):
    t2 = np.ascontiguousarray(np.random.rand(no, no, nv, nv))
    w = np.ascontiguousarray(np.random.rand(nv, no, nv, no))

    m = int(t2.size * frac)

    # Keep the tests small
    if m > 200:
        print(f"M={m}")
        raise ValueError("M >= 1000, choose a smaller matrix or a smaller fraction.")

    # Compress by getting the largest m elements
    t2_compressed = SparseTensor4d(t2.ravel(), t2.shape, m, "largest")

    # Zero out the elements of the original t2 array
    idx = np.unravel_index(np.argsort(t2, axis=None), t2.shape)
    for i in range(t2.size - m):
        t2[idx[0][i], idx[1][i], idx[2][i], idx[3][i]] = 0

    t2_new_np = -1 * np.einsum("akci,kjcb->ijab", w, t2, order="C")
    t2_new_fri = np.zeros(t2.shape, order="C")
    contract_DTSpT(w.ravel(), t2_compressed, t2_new_fri.ravel(), "1202")

    npt.assert_almost_equal(t2_new_fri, t2_new_np)


@pytest.mark.parametrize("no,nv,frac", [(4, 8, 0.1), (5, 10, 0.01), (10, 20, 0.001)])
def test_1303_contraction(no, nv, frac):
    t2 = np.ascontiguousarray(np.random.rand(no, no, nv, nv))
    w = np.ascontiguousarray(np.random.rand(nv, no, no, nv))

    m = int(t2.size * frac)

    # Keep the tests small
    if m > 200:
        print(f"M={m}")
        raise ValueError("M >= 1000, choose a smaller matrix or a smaller fraction.")

    # Compress by getting the largest m elements
    t2_compressed = SparseTensor4d(t2.ravel(), t2.shape, m, "largest")

    # Zero out the elements of the original t2 array
    idx = np.unravel_index(np.argsort(t2, axis=None), t2.shape)
    for i in range(t2.size - m):
        t2[idx[0][i], idx[1][i], idx[2][i], idx[3][i]] = 0

    t2_new_np = np.einsum("akic,kjbc->ijab", w, t2, order="C")
    t2_new_fri = np.zeros(t2.shape, order="C")
    contract_DTSpT(w.ravel(), t2_compressed, t2_new_fri.ravel(), "1303")

    npt.assert_almost_equal(t2_new_fri, t2_new_np)


@pytest.mark.parametrize("no,nv,frac", [(4, 8, 0.1), (5, 10, 0.01), (10, 20, 0.001)])
def test_1203_contraction(no, nv, frac):
    t2 = np.ascontiguousarray(np.random.rand(no, no, nv, nv))
    w = np.ascontiguousarray(np.random.rand(nv, no, nv, no))

    m = int(t2.size * frac)

    # Keep the tests small
    if m > 200:
        print(f"M={m}")
        raise ValueError("M >= 1000, choose a smaller matrix or a smaller fraction.")

    # Compress by getting the largest m elements
    t2_compressed = SparseTensor4d(t2.ravel(), t2.shape, m, "largest")

    # Zero out the elements of the original t2 array
    idx = np.unravel_index(np.argsort(t2, axis=None), t2.shape)
    for i in range(t2.size - m):
        t2[idx[0][i], idx[1][i], idx[2][i], idx[3][i]] = 0

    t2_new_np = np.einsum("bkci,kjac->ijab", w, t2, order="C")
    t2_new_fri = np.zeros(t2.shape, order="C")
    contract_DTSpT(w.ravel(), t2_compressed, t2_new_fri.ravel(), "1203")

    npt.assert_almost_equal(t2_new_fri, t2_new_np)


@pytest.mark.parametrize("no,nv,frac", [(4, 8, 0.1), (5, 10, 0.01), (10, 20, 0.001)])
def test_1323_contraction(no, nv, frac):
    t2 = np.ascontiguousarray(np.random.rand(no, no, nv, nv))
    ovov = np.ascontiguousarray(np.random.rand(no, nv, no, nv))

    m = int(t2.size * frac)

    # Keep the tests small
    if m > 200:
        print(f"M={m}")
        raise ValueError("M >= 1000, choose a smaller matrix or a smaller fraction.")

    # Compress by getting the largest m elements
    t2_compressed = SparseTensor4d(t2.ravel(), t2.shape, m, "largest")

    # Zero out the elements of the original t2 array
    idx = np.unravel_index(np.argsort(t2, axis=None), t2.shape)
    for i in range(t2.size - m):
        t2[idx[0][i], idx[1][i], idx[2][i], idx[3][i]] = 0

    Wklij_np = np.einsum("kcld,ijcd->klij", ovov, t2, order="C")
    Wklij = np.zeros(Wklij_np.shape, order="C")
    contract_DTSpT(ovov.ravel(), t2_compressed, Wklij.ravel(), "1323")

    npt.assert_almost_equal(Wklij, Wklij_np)


@pytest.mark.parametrize("no,nv,frac", [(4, 8, 0.1), (5, 10, 0.01), (10, 20, 0.001)])
def test_0112_contraction(no, nv, frac):
    t2 = np.ascontiguousarray(np.random.rand(no, no, nv, nv))
    ovov = np.ascontiguousarray(np.random.rand(no, nv, no, nv))

    m = int(t2.size * frac)

    # Keep the tests small
    if m > 200:
        print(f"M={m}")
        raise ValueError("M >= 1000, choose a smaller matrix or a smaller fraction.")

    # Compress by getting the largest m elements
    t2_compressed = SparseTensor4d(t2.ravel(), t2.shape, m, "largest")

    # Zero out the elements of the original t2 array
    idx = np.unravel_index(np.argsort(t2, axis=None), t2.shape)
    for i in range(t2.size - m):
        t2[idx[0][i], idx[1][i], idx[2][i], idx[3][i]] = 0

    Wakic_np = -0.5 * np.einsum("ldkc,ilda->akic", ovov, t2, order="C")
    Wakic = np.zeros(Wakic_np.shape, order="C")
    contract_DTSpT(ovov.ravel(), t2_compressed, Wakic.ravel(), "0112")

    npt.assert_almost_equal(Wakic, Wakic_np)


@pytest.mark.parametrize("no,nv,frac", [(4, 8, 0.1), (5, 10, 0.01), (10, 20, 0.001)])
def test_0113_contraction(no, nv, frac):
    t2 = np.ascontiguousarray(np.random.rand(no, no, nv, nv))
    ovov = np.ascontiguousarray(np.random.rand(no, nv, no, nv))

    m = int(t2.size * frac)

    # Keep the tests small
    if m > 200:
        print(f"M={m}")
        raise ValueError("M >= 1000, choose a smaller matrix or a smaller fraction.")

    # Compress by getting the largest m elements
    t2_compressed = SparseTensor4d(t2.ravel(), t2.shape, m, "largest")

    # Zero out the elements of the original t2 array
    idx = np.unravel_index(np.argsort(t2, axis=None), t2.shape)
    for i in range(t2.size - m):
        t2[idx[0][i], idx[1][i], idx[2][i], idx[3][i]] = 0

    Wakic_np = np.einsum("ldkc,ilad->akic", ovov, t2, order="C")
    Wakic = np.zeros(Wakic_np.shape, order="C")
    contract_DTSpT(ovov.ravel(), t2_compressed, Wakic.ravel(), "0113")

    npt.assert_almost_equal(Wakic, Wakic_np)


@pytest.mark.parametrize("no,nv,frac", [(4, 8, 0.1), (5, 10, 0.01), (10, 20, 0.001)])
def test_0313_contraction(no, nv, frac):
    t2 = np.ascontiguousarray(np.random.rand(no, no, nv, nv))
    ovov = np.ascontiguousarray(np.random.rand(no, nv, no, nv))

    m = int(t2.size * frac)

    # Keep the tests small
    if m > 200:
        print(f"M={m}")
        raise ValueError("M >= 1000, choose a smaller matrix or a smaller fraction.")

    # Compress by getting the largest m elements
    t2_compressed = SparseTensor4d(t2.ravel(), t2.shape, m, "largest")

    # Zero out the elements of the original t2 array
    idx = np.unravel_index(np.argsort(t2, axis=None), t2.shape)
    for i in range(t2.size - m):
        t2[idx[0][i], idx[1][i], idx[2][i], idx[3][i]] = 0

    Wakic_np = -0.5 * np.einsum("lckd,ilad->akic", ovov, t2, order="C")
    Wakic = np.zeros(Wakic_np.shape, order="C")
    contract_DTSpT(ovov.ravel(), t2_compressed, Wakic.ravel(), "0313")

    npt.assert_almost_equal(Wakic, Wakic_np)


@pytest.mark.parametrize("no,nv,frac", [(4, 8, 0.1), (5, 10, 0.01), (10, 20, 0.001)])
def test_0312_contraction(no, nv, frac):
    t2 = np.ascontiguousarray(np.random.rand(no, no, nv, nv))
    ovov = np.ascontiguousarray(np.random.rand(no, nv, no, nv))

    m = int(t2.size * frac)

    # Keep the tests small
    if m > 200:
        print(f"M={m}")
        raise ValueError("M >= 1000, choose a smaller matrix or a smaller fraction.")

    # Compress by getting the largest m elements
    t2_compressed = SparseTensor4d(t2.ravel(), t2.shape, m, "largest")

    # Zero out the elements of the original t2 array
    idx = np.unravel_index(np.argsort(t2, axis=None), t2.shape)
    for i in range(t2.size - m):
        t2[idx[0][i], idx[1][i], idx[2][i], idx[3][i]] = 0

    Wakci_np = -0.5 * np.einsum("lckd,ilda->akci", ovov, t2, order="C")
    Wakci = np.zeros(Wakci_np.shape, order="C")
    contract_DTSpT(ovov.ravel(), t2_compressed, Wakci.ravel(), "0312")

    npt.assert_almost_equal(Wakci, Wakci_np)
