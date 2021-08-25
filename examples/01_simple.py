import time
from pyscf import gto, scf, cc
from pyscf.cc import ccsd
import fricc

ethane = """H      1.1851     -0.0039      0.9875
C      0.7516     -0.0225     -0.0209
H      1.1669      0.8330     -0.5693
H      1.1155     -0.9329     -0.5145
C     -0.7516      0.0225      0.0209
H     -1.1669     -0.8334      0.5687
H     -1.1157      0.9326      0.5151
H     -1.1850      0.0044     -0.9875"""

benzene = """  H      1.2194     -0.1652      2.1600
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
  H      2.4836     -0.1022      0.0205"""


mol = gto.M(atom=benzene, basis="sto3g", verbose=4)

nocc = mol.nelec[0]
nvirt = mol.nao_nr() - nocc
print(f" nocc = {nocc}\nnvirt = {nvirt}")
print(f"|t2| = {nocc**2 * nvirt**2}")

mf = scf.RHF(mol).run()


mycc2 = cc.CCSD(mf)
t_ccsd = time.time()
mycc2.kernel()
t_ccsd = time.time() - t_ccsd

# User some custom functions instead of the default PySCF ones
ccsd.CCSD.update_amps = fricc.update_amps
# ccsd.CCSD.kernel = fricc.kernel

fri_settings = {
    "m_keep": 99225,
    "compression": "fri",
    "sampling_method": "systematic",
    "verbose": True if mol.verbose >= 5 else False,
}
mycc = cc.CCSD(mf)
mycc.diis_start_cycle = mycc.max_cycle + 1
mycc.fri_settings = fri_settings
mycc.max_cycle = 20

t_fricc = time.time()
mycc.kernel()
t_fricc = time.time() - t_fricc

print(f"Error {abs(mycc.e_tot-mycc2.e_tot):.2e}")
print(f"CCSD     Time {t_ccsd:.1f}")
print(f"FRI-CCSD Time {t_fricc:.1f}")
# print(mycc.energies)
print(f" nocc = {nocc}\nnvirt = {nvirt}")
print(f"|t2| = {nocc**2 * nvirt**2}")
