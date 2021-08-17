import time
import numpy as np
from fricc import get_m_largest, partial_argsort, partial_argsort_paired

np.random.seed(20)

n = int(4e8)
frac = 0.00000001
m = int(n * frac)
print(f"Size of   t2  = {n}")
print(f"Size of \u03A6[t2] = {m}")

v1 = np.random.random(n).astype(np.float64)

# t0 = time.time()
# np_idx = np.flip(np.argsort(v1))[:m]
# print(f"NumPy Sorting {time.time()-t0:.3f}")

fricc_idx = np.zeros(m, dtype=np.uint64)

# t0 = time.time()
# get_m_largest(v1, m, fricc_idx)
# print(f"FRI Full Sorting {time.time()-t0:.3f}")

# t0 = time.time()
# fricc_idx = partial_argsort(v1, m)
# print(f"FRI Partial Sorting {time.time()-t0:.3f}")

t0 = time.time()
fricc_idx = partial_argsort_paired(v1, m)
print(f"FRI Partial Paired Sorting {time.time()-t0:.3f}")
