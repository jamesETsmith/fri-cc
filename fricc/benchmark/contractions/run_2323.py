import time
import numpy as np
from fricc.py_rccsd import SparseTensor4d, contract_DTSpT, contract_DTSpT_experimental

no = 50
nv = 150

t2 = np.ascontiguousarray(np.random.rand(no, no, nv, nv))
w = np.ascontiguousarray(np.random.rand(nv, nv, nv, nv))

t0 = time.time()
t2_new_np = np.einsum("abcd,ijcd->ijab", w, t2, order="C", optimize=True)
print(f"Time for numpy contraction           {time.time()-t0:.3f}")


for frac in [1e-2, 1e-3, 1e-4]:
    m = int(t2.size * frac)
    print(f"Size of   t2  = {t2.size}")
    print(f"Size of \u03A6[t2] = {m}")

    # Compress by getting the largest m elements
    t2_compressed = SparseTensor4d(t2, t2.shape, m)
    t2_new_fri = np.zeros(t2.shape, order="C")

    t0 = time.time()
    contract_DTSpT(w, t2_compressed, t2_new_fri, "2323")
    print(f"Time for FRI-CC contraction          {time.time()-t0:.3f}")

    mods = ["taskloop", "atomic"]
    for m in mods:
        t0 = time.time()
        contract_DTSpT_experimental(w, t2_compressed, t2_new_fri, m)
        print(f"Time for FRI-CC {m:8s} contraction {time.time()-t0:.3f}")

    print()