from pyscf import gto, scf, cc
import fricc

# From So Hirata http://faculty.scs.illinois.edu/hirata/cc_data.out
# mol = gto.M(
#     atom="""H    0.000000000000000   1.079252144093028   1.474611055780858
#             O    0.000000000000000   0.000000000000000   0.000000000000000
#             H    0.000000000000000   1.079252144093028  -1.474611055780858""",
#     basis="""H S
#               3.42525091         0.15432897
#               0.62391373         0.53532814
#               0.16885540         0.44463454
# O S
#             130.70932000         0.15432897
#              23.80886100         0.53532814
#               6.44360830         0.44463454
# O S
#               5.03315130        -0.09996723
#               1.16959610         0.39951283
#               0.38038900         0.70011547
# O P
#               5.03315130         0.15591627
#               1.16959610         0.60768372
#               0.38038900         0.39195739""",
#     verbose=3,
#     unit="B",
# )
mol = gto.M(atom="H 0 0 0; H 0 0 0.743", basis="aug-cc-pvtz", verbose=3)
# mol = gto.M(atom="H 0 0 0;", basis="aug-cc-pvtz", charge=-1, verbose=3)
# mol = gto.M(atom="He 0 0 0", basis="aug-cc-pvtz", verbose=3)

CONV_TOL = 1e-9
mf = scf.RHF(mol).run()


myccsd = cc.CCSD(mf)
myccsd.conv_tol = CONV_TOL
myccsd.kernel()


mylccsd = fricc.FRILCCSD(
    mf,
    fri_settings={
        "compressed_contractions": [],
        "compression": "largest",
        "m_keep": 10,
    },
)
mylccsd.conv_tol = CONV_TOL
# mylccsd.frozen = 2
mylccsd.kernel()

print(f"Error (Ha) {mylccsd.e_tot-myccsd.e_tot}")
