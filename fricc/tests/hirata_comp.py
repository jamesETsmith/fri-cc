from pyscf import gto, scf, cc
import fricc

data={"mp2":-0.0358672469, "ccsd":-0.0501273286, "lccsd":-0.0508915694}

mol = gto.M(atom="""H    0.000000000000000   1.079252144093028   1.474611055780858
O    0.000000000000000   0.000000000000000   0.000000000000000
H    0.000000000000000   1.079252144093028  -1.474611055780858""", unit="au", basis="""
H S   
              3.42525091         0.15432897
              0.62391373         0.53532814
              0.16885540         0.44463454
O S   
            130.70932000         0.15432897
             23.80886100         0.53532814
              6.44360830         0.44463454
O S   
              5.03315130        -0.09996723
              1.16959610         0.39951283
              0.38038900         0.70011547
O P   
              5.03315130         0.15591627
              1.16959610         0.60768372
              0.38038900         0.39195739
""", verbose=3, symmetry=True)

mf = scf.RHF(mol).newton().run()

mycc = cc.CCSD(mf)
mycc.frozen = 2
mycc.kernel()

mp2_err = abs(data["mp2"] - mycc.emp2)
ccsd_err = abs(data["ccsd"] - mycc.e_corr)
print(f"MP2 error  {mp2_err:.4f} (Ha)")
print(f"CCSD error {ccsd_err:.4f} (Ha)")
