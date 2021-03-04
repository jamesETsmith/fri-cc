import os
import time
import pandas as pd
from pyscf import gto, scf, cc
from pyscf.cc.rccsd import update_amps as update_amps_rccsd
from fricc import update_amps_wrapper

#
# User settings
#

for d in ["_data", "_logs"]:
    os.makedirs(d, exist_ok=True)


data = {
    "OMP_NUM_THREADS": [],
    "PySCF Time (s)": [],
    "PySCF (RCCSD) Time (s)": [],
    "FRICC Time (s)": [],
}

# Setup
# os.environ["OMP_NUM_THREADS"] = "4"
mol = gto.M(
    atom="""  H      1.2194     -0.1652      2.1600
  C      0.6825     -0.0924      1.2087
  C     -0.7075     -0.0352      1.1973
  H     -1.2644     -0.0630      2.1393
  C     -1.3898      0.0572     -0.0114
  H     -2.4836      0.1021     -0.0204
  C     -0.6824      0.0925     -1.2088
  H     -1.2194      0.1652     -2.1599
  C      0.7075      0.0352     -1.1973
  H      1.2641      0.0628     -2.1395
  C      1.3899     -0.0572      0.0114
  H      2.4836     -0.1022      0.0205""",
    basis="ccpvdz",
)
mf = scf.RHF(mol).run()
mycc = cc.CCSD(mf)
mycc.max_cycle = 1
mycc.kernel()

# Save the intermediates
t1 = mycc.t1
t2 = mycc.t2
eris = mycc.ao2mo()
nmo = len(mf.mo_energy)
print(f"N = {nmo}\tN^6 = {nmo **6:.1e}")

#
# Benchmarking
#

nthreads_list = [1, 2, 4, 8, 16, 24]
nthreads_list = [16]

for nthreads in nthreads_list:
    os.environ["OMP_NUM_THREADS"] = f"{nthreads}"
    print(f"Running Benchmark for OMP_NUM_THREADS={nthreads}")

    ti_pyscf = time.time()
    mycc.update_amps(t1, t2, eris)
    tf_pyscf = time.time() - ti_pyscf
    print(f"PySCF update_amps time {tf_pyscf:.3f} (s)")

    ti_pyscf2 = time.time()
    update_amps_rccsd(mycc, t1, t2, eris)
    tf_pyscf2 = time.time() - ti_pyscf2
    print(f"PySCF update_amps (from rccsd.py) time {tf_pyscf2:.3f} (s)")

    ti_fricc = time.time()
    update_amps_wrapper(t1, t2, eris)
    tf_fricc = time.time() - ti_fricc
    print(f"PySCF update_amps time {tf_fricc:.3f} (s)\n")

    # Save data
    data["OMP_NUM_THREADS"].append(nthreads)
    data["PySCF Time (s)"].append(tf_pyscf)
    data["PySCF (RCCSD) Time (s)"].append(tf_pyscf2)
    data["FRICC Time (s)"].append(tf_fricc)

df = pd.DataFrame(data)
print(df)