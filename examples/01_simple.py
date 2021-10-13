import time
from pyscf import gto, scf, cc
from pyscf.cc import ccsd
import fricc


pentane = """C          1.07644        0.16055       -0.12068
C          2.59276        0.16199       -0.00016
C          3.08852       -1.00227        0.85812
C          4.61180       -0.99654        0.97485
C          5.10826       -2.15033        1.83143
H          0.74503        1.00196       -0.73569
H          0.60562        0.25114        0.86269
H          0.72214       -0.76329       -0.58840
H          2.91753        1.11305        0.43862
H          3.03329        0.10015       -1.00255
H          2.75629       -1.94991        0.41784
H          2.64207       -0.93655        1.85776
H          4.94874       -0.05141        1.41564
H          5.06238       -1.06929       -0.02136
H          4.70175       -2.08942        2.84660
H          6.20095       -2.12814        1.90186
H          4.81547       -3.11370        1.40130"""
# Pentane with ccpvdz fails with m_keep = 1e5
# Pentane with ccpvdz fails with m_keep = 1e6


mol = gto.M(atom=pentane, basis="ccpvdz", verbose=4)

nocc = mol.nelec[0]
nvirt = mol.nao_nr() - nocc
print(f" nocc = {nocc}\nnvirt = {nvirt}")
print(f"|t2| = {nocc**2 * nvirt**2}")

mf = scf.RHF(mol).run()


# mycc2 = cc.CCSD(mf)
# t_ccsd = time.time()
# mycc2.kernel()
# t_ccsd = time.time() - t_ccsd

fri_settings = {
    "m_keep": 1e4,
    "compression": "fri",
    "sampling_method": "systematic",
    # "compressed_contractions": ["O^2V^4"],
    # "compressed_contractions": [],
}
mycc = fricc.FRICCSD(mf, fri_settings=fri_settings)
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
