import numpy as np
from fricc import get_m_largest
import pytest


@pytest.mark.parametrize("m, n", [(10, 100), (50, 100), (100, 100)])
def test_get_m_largest(m, n):
    v1 = np.random.random(n).astype(np.float64)
    np_idx = np.argsort(v1)[:m]
    fricc_idx = np.zeros(m, dtype=np.uint64)
    get_m_largest(v1, m, fricc_idx)
    np.testing.assert_array_equal(np_idx, fricc_idx)
