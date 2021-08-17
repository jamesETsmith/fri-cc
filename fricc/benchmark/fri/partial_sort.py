import time
import numpy as np
from fricc import parallel_partial_sort

np.random.seed(20)

n = int(4e8)
frac = 0.0001
m = int(n * frac)
print(f"Size of   t2  = {n}")
print(f"Size of \u03A6[t2] = {m}")

v1 = np.random.random(n).astype(np.float64)

# t0 = time.time()
# np_idx = np.flip(np.argsort(v1))[:m]
# print(f"NumPy Sorting {time.time()-t0:.3f}")


t0 = time.time()
idx = parallel_partial_sort(v1, m)
print(f"FRI Sorting {time.time()-t0:.3f}")
