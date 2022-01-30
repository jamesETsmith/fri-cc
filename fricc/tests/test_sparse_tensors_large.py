import numpy as np
import pytest
from fricc.py_rccsd import SparseTensor4d, contract_DTSpT

npt = np.testing
np.random.seed(20)

def create_sparse_ndarray(t_shape, nnz):
    size = np.prod(t_shape)
    tensor = np.zeros(size, order="C")

    idx = np.random.choice(np.arange(size), size=int(nnz))
    tensor[idx] = np.random.rand(int(nnz))
    
    return tensor.reshape(t_shape)


@pytest.mark.parametrize(
    "no,nv,frac",
    [
        (20,40,1e-4),
        (50, 90, 1e-6),
    ],
)
def test_2323_contraction(no, nv, frac):
    w = np.random.rand(nv, nv, nv, nv)

    t_shape = (no,no,nv,nv)
    nnz = int(np.prod(t_shape)*frac)
    t2 = create_sparse_ndarray(t_shape, nnz)

    m = int(t2.size * frac)

    # Compress by getting the largest m elements
    t2_compressed = SparseTensor4d(t2, t2.shape, m, "largest")


    # t2new += lib.einsum('abcd,ijcd->ijab', Wvvvv, tau)
    t2_new_np = np.einsum("abcd,ijcd->ijab", w, t2, order="C")
    t2_new_fri = np.zeros(t2.shape, order="C")
    contract_DTSpT(w, t2_compressed, t2_new_fri, "2323")
    npt.assert_almost_equal(t2_new_fri, t2_new_np)
