import time
import numpy as np
from fricc import parallel_sort

np.random.seed(20)

n = int(1e8)

v1 = np.random.random(n).astype(np.float64)

t0 = time.time()
np.sort(v1)
print(f"NumPy Sorting {time.time()-t0:.3f}")


t0 = time.time()
parallel_sort(v1)
print(f"FRI Sorting {time.time()-t0:.3f}")
