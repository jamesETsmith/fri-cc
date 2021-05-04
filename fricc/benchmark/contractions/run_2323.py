import time
import numpy as np
from fricc import SparseTensor4d, contract_DTSpT

no = 100
nv = 200
frac = 0.0001

t2 = np.ascontiguousarray(np.random.rand(no, no, nv, nv))
w = np.ascontiguousarray(np.random.rand(nv, nv, nv, nv))

m = int(t2.size * frac)
print(f"Size of   t2  = {t2.size}")
print(f"Size of \u03A6[t2] = {m}")


t0 = time.time()
t2_new_np = np.einsum("abcd,ijcd->ijab", w, t2, order="C", optimize=True)
print(f"Time for numpy contraction  {time.time()-t0:.3f}")


# Compress by getting the largest m elements
t2_compressed = SparseTensor4d(t2.ravel(), t2.shape, m)
t2_new_fri = np.zeros(t2.shape, order="C")

t0 = time.time()
contract_DTSpT(w.ravel(), t2_compressed, t2_new_fri.ravel(), "2323")
print(f"Time for FRI-CC contraction {time.time()-t0:.3f}")
