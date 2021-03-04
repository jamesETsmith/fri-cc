#include <rccsd.hpp>

namespace RCCSD {
void update_amps(
    Eigen::Ref<RowMatrixXd> t1_mat, Eigen::Ref<Eigen::VectorXd> t2_vec,
    Eigen::Ref<RowMatrixXd> t1_mat_new, Eigen::Ref<Eigen::VectorXd> t2_vec_new,
    Eigen::Ref<RowMatrixXd> fock_oo_mat, Eigen::Ref<RowMatrixXd> fock_ov_mat,
    Eigen::Ref<RowMatrixXd> fock_vv_mat, Eigen::Ref<Eigen::VectorXd> oooo_vec,
    Eigen::Ref<Eigen::VectorXd> ovoo_vec, Eigen::Ref<Eigen::VectorXd> oovv_vec,
    Eigen::Ref<Eigen::VectorXd> ovvo_vec, Eigen::Ref<Eigen::VectorXd> ovov_vec,
    Eigen::Ref<Eigen::VectorXd> ovvv_vec, Eigen::Ref<Eigen::VectorXd> vvvv_vec,
    Eigen::Ref<Eigen::VectorXd> mo_energies) {

  // Helpers
  Eigen::array<Eigen::IndexPair<int>, 0> contraction_dims_0d = {};
  Eigen::array<Eigen::IndexPair<int>, 1> contraction_dims_1d;
  Eigen::array<Eigen::IndexPair<int>, 2> contraction_dims_2d;
  Eigen::array<Eigen::IndexPair<int>, 3> contraction_dims_3d;
  Eigen::array<int, 2> shuffle_idx_2d;
  Eigen::array<int, 4> shuffle_idx_4d;
  RowTensor2d tmp_2d;
  RowTensor4d tmp_4d;
  shuffle_idx_2d = {1, 0};

  //
  // Unpack input args and map to tensors
  //

  const int nocc = t1_mat.rows(), nvirt = t1_mat.cols();
  TMap2d t1(t1_mat.data(), nocc, nvirt);
  TMap4d t2(t2_vec.data(), nocc, nocc, nvirt, nvirt);
  TMap2d t1_new(t1_mat_new.data(), nocc, nvirt);
  TMap4d t2_new(t2_vec_new.data(), nocc, nocc, nvirt, nvirt);

  TMap2d fock_oo(fock_oo_mat.data(), nocc, nocc);
  TMap2d fock_ov(fock_ov_mat.data(), nocc, nvirt);
  TMap2d fock_vv(fock_vv_mat.data(), nvirt, nvirt);

  TMap4d oooo(oooo_vec.data(), nocc, nocc, nocc, nocc);
  TMap4d ovoo(ovoo_vec.data(), nocc, nvirt, nocc, nocc);

  TMap4d oovv(oovv_vec.data(), nocc, nocc, nvirt, nvirt);
  TMap4d ovvo(ovvo_vec.data(), nocc, nvirt, nvirt, nocc);
  TMap4d ovov(ovov_vec.data(), nocc, nvirt, nocc, nvirt);

  TMap4d ovvv(ovvv_vec.data(), nocc, nvirt, nvirt, nvirt);
  TMap4d vvvv(vvvv_vec.data(), nvirt, nvirt, nvirt, nvirt);

  // Move energy terms to the other side
  RowTensor2d Foo(nocc, nocc);
  RowTensor2d Fov(nocc, nvirt);
  RowTensor2d Fvv(nvirt, nvirt);
  Foo.setZero(), Fov.setZero(), Fvv.setZero();

  Eigen::Map<RowMatrixXd> Foo_mat(Foo.data(), nocc, nocc);
  Eigen::Map<RowMatrixXd> Fov_mat(Fov.data(), nocc, nvirt);
  Eigen::Map<RowMatrixXd> Fvv_mat(Fvv.data(), nvirt, nvirt);

  auto f_int_start_time = std::chrono::steady_clock::now();
  make_Foo(t1_mat, t2_vec, fock_oo_mat, ovov_vec, Foo_mat);
  make_Fov(t1_mat, t2_vec, fock_ov_mat, ovov_vec, Fov_mat);
  make_Fvv(t1_mat, t2_vec, fock_vv_mat, ovov_vec, Fvv_mat);
  auto f_int_end_time = std::chrono::steady_clock::now();
  std::chrono::duration<double> f_int_elapsed = f_int_end_time - f_int_start_time; // in seconds
  std::cout << "Time to create F intermediates " << f_int_elapsed.count() << " (s)" << std::endl;

  Eigen::VectorXd mo_e_o = mo_energies.head(nocc);
  Eigen::VectorXd mo_e_v = mo_energies.tail(nvirt);
  Foo_mat.diagonal() = Foo_mat.diagonal() - mo_e_o;
  Fvv_mat.diagonal() = Fvv_mat.diagonal() - mo_e_v;

  //
  // T1 update
  //

  // -2*np.einsum('kc,ka,ic->ia', fov, t1, t1)
  contraction_dims_1d = {Eigen::IndexPair<int>(0, 0)};
  tmp_2d = fock_ov.contract(t1, contraction_dims_1d);
  contraction_dims_1d = {Eigen::IndexPair<int>(0, 1)};
  t1_new -=
      2 * tmp_2d.contract(t1, contraction_dims_1d).shuffle(shuffle_idx_2d);

  // np.einsum('ac,ic->ia', Fvv, t1)
  contraction_dims_1d = {Eigen::IndexPair<int>(1, 1)};
  t1_new += Fvv.contract(t1, contraction_dims_1d).shuffle(shuffle_idx_2d);

  // -np.einsum('ki,ka->ia', Foo, t1)
  contraction_dims_1d = {Eigen::IndexPair<int>(0, 0)};
  t1_new -= Foo.contract(t1, contraction_dims_1d);

  // t1new += 2*np.einsum('kc,kica->ia', Fov, t2)
  contraction_dims_2d = {Eigen::IndexPair<int>(0, 0),
                         Eigen::IndexPair<int>(1, 2)};
  t1_new += 2 * Fov.contract(t2, contraction_dims_2d);

  // t1new +=  -np.einsum('kc,ikca->ia', Fov, t2)
  contraction_dims_2d = {Eigen::IndexPair<int>(0, 1),
                         Eigen::IndexPair<int>(1, 2)};
  t1_new -= Fov.contract(t2, contraction_dims_2d);

  // t1new +=   np.einsum('kc,ic,ka->ia', Fov, t1, t1)
  contraction_dims_1d = {Eigen::IndexPair<int>(1, 1)};
  tmp_2d = Fov.contract(t1, contraction_dims_1d);
  contraction_dims_1d = {Eigen::IndexPair<int>(0, 0)};
  t1_new += tmp_2d.contract(t1, contraction_dims_1d);

  // t1new += fov.conj()
  t1_new += fock_ov;

  // t1new += 2*np.einsum('kcai,kc->ia', eris.ovvo, t1)
  contraction_dims_2d = {Eigen::IndexPair<int>(0, 0),
                         Eigen::IndexPair<int>(1, 1)};
  t1_new += 2 * ovvo.contract(t1, contraction_dims_2d).shuffle(shuffle_idx_2d);

  // t1new +=  -np.einsum('kiac,kc->ia', eris.oovv, t1)
  contraction_dims_2d = {Eigen::IndexPair<int>(0, 0),
                         Eigen::IndexPair<int>(3, 1)};
  t1_new -= oovv.contract(t1, contraction_dims_2d);

  // t1new += 2*lib.einsum('kdac,ikcd->ia', eris_ovvv, t2)
  contraction_dims_3d = {Eigen::IndexPair<int>(0, 1),
                         Eigen::IndexPair<int>(1, 3),
                         Eigen::IndexPair<int>(3, 2)};
  t1_new += 2 * ovvv.contract(t2, contraction_dims_3d).shuffle(shuffle_idx_2d);

  // t1new +=  -lib.einsum('kcad,ikcd->ia', eris_ovvv, t2)
  contraction_dims_3d = {Eigen::IndexPair<int>(0, 1),
                         Eigen::IndexPair<int>(1, 2),
                         Eigen::IndexPair<int>(3, 3)};
  t1_new -= ovvv.contract(t2, contraction_dims_3d).shuffle(shuffle_idx_2d);

  // t1new += 2*lib.einsum('kdac,kd,ic->ia', eris_ovvv, t1, t1)
  contraction_dims_2d = {Eigen::IndexPair<int>(0, 0),
                         Eigen::IndexPair<int>(1, 1)};
  tmp_2d = 2 * ovvv.contract(t1, contraction_dims_2d);
  contraction_dims_1d = {Eigen::IndexPair<int>(1, 1)};
  t1_new += tmp_2d.contract(t1, contraction_dims_1d).shuffle(shuffle_idx_2d);

  // t1new +=  -lib.einsum('kcad,kd,ic->ia', eris_ovvv, t1, t1)
  contraction_dims_2d = {Eigen::IndexPair<int>(0, 0),
                         Eigen::IndexPair<int>(3, 1)};
  tmp_2d = ovvv.contract(t1, contraction_dims_2d);
  contraction_dims_1d = {Eigen::IndexPair<int>(0, 1)};
  t1_new -= tmp_2d.contract(t1, contraction_dims_1d).shuffle(shuffle_idx_2d);

  // t1new += -2 * lib.einsum("lcki,klac->ia", eris_ovoo, t2)
  contraction_dims_3d = {Eigen::IndexPair<int>(0, 1),
                         Eigen::IndexPair<int>(1, 3),
                         Eigen::IndexPair<int>(2, 0)};
  t1_new -= 2 * ovoo.contract(t2, contraction_dims_3d);

  // t1new += lib.einsum("kcli,klac->ia", eris_ovoo, t2)
  contraction_dims_3d = {Eigen::IndexPair<int>(0, 0),
                         Eigen::IndexPair<int>(1, 3),
                         Eigen::IndexPair<int>(2, 1)};
  t1_new += ovoo.contract(t2, contraction_dims_3d);

  // t1new += -2 * lib.einsum("lcki,lc,ka->ia", eris_ovoo, t1, t1)
  contraction_dims_2d = {Eigen::IndexPair<int>(0, 0),
                         Eigen::IndexPair<int>(1, 1)};
  tmp_2d = 2 * ovoo.contract(t1, contraction_dims_2d);
  contraction_dims_1d = {Eigen::IndexPair<int>(0, 0)};
  t1_new -= tmp_2d.contract(t1, contraction_dims_1d);

  // t1new += lib.einsum("kcli,lc,ka->ia", eris_ovoo, t1, t1)
  contraction_dims_2d = {Eigen::IndexPair<int>(2, 0),
                         Eigen::IndexPair<int>(1, 1)};
  tmp_2d = ovoo.contract(t1, contraction_dims_2d);
  contraction_dims_1d = {Eigen::IndexPair<int>(0, 0)};
  t1_new += tmp_2d.contract(t1, contraction_dims_1d);

  //
  // T2 update
  //

  // tmp2  = lib.einsum('kibc,ka->abic', eris.oovv, -t1)
  contraction_dims_1d = {Eigen::IndexPair<int>(0, 0)};
  shuffle_idx_4d = {3, 1, 0, 2};
  tmp_4d = oovv.contract(-t1, contraction_dims_1d).shuffle(shuffle_idx_4d);

  // tmp2 += np.asarray(eris_ovvv).conj().transpose(1,3,0,2)
  shuffle_idx_4d = {1, 3, 0, 2};
  tmp_4d += ovvv.shuffle(shuffle_idx_4d);

  // tmp = lib.einsum('abic,jc->ijab', tmp2, t1)
  contraction_dims_1d = {Eigen::IndexPair<int>(3, 1)};
  shuffle_idx_4d = {2, 3, 0, 1};
  RowTensor4d tmp_4d_2;
  tmp_4d_2 = tmp_4d.contract(t1, contraction_dims_1d).shuffle(shuffle_idx_4d);

  // t2new = tmp + tmp.transpose(1,0,3,2)
  shuffle_idx_4d = {1, 0, 3, 2};
  t2_new += tmp_4d_2 + tmp_4d_2.shuffle(shuffle_idx_4d);

  // tmp2  = lib.einsum('kcai,jc->akij', eris.ovvo, t1)
  contraction_dims_1d = {Eigen::IndexPair<int>(1, 1)};
  shuffle_idx_4d = {1, 0, 2, 3};
  tmp_4d = ovvo.contract(t1, contraction_dims_1d).shuffle(shuffle_idx_4d);

  // tmp2 += eris_ovoo.transpose(1,3,0,2).conj()
  shuffle_idx_4d = {1, 3, 0, 2};
  tmp_4d += ovoo.shuffle(shuffle_idx_4d);

  // tmp = lib.einsum('akij,kb->ijab', tmp2, t1)
  contraction_dims_1d = {Eigen::IndexPair<int>(1, 0)};
  shuffle_idx_4d = {1, 2, 0, 3};
  tmp_4d_2 = tmp_4d.contract(t1, contraction_dims_1d).shuffle(shuffle_idx_4d);

  // t2new -= tmp + tmp.transpose(1,0,3,2)
  shuffle_idx_4d = {1, 0, 3, 2};
  t2_new -= tmp_4d_2 + tmp_4d_2.shuffle(shuffle_idx_4d);

  // t2new += np.asarray(eris.ovov).conj().transpose(0,2,1,3)
  shuffle_idx_4d = {0, 2, 1, 3};
  t2_new += ovov.shuffle(shuffle_idx_4d);

  // Using the intermediates

  RowTensor2d Loo(nocc, nocc), Lvv(nvirt, nvirt);
  Loo.setZero(), Lvv.setZero();
  Eigen::Map<RowMatrixXd> Loo_mat(Loo.data(), nocc, nocc);
  Eigen::Map<RowMatrixXd> Lvv_mat(Lvv.data(), nvirt, nvirt);

  auto l_int_start_time = std::chrono::steady_clock::now();
  make_Loo(t1_mat, t2_vec, fock_ov_mat, ovoo_vec, Foo_mat, Loo_mat);
  make_Lvv(t1_mat, t2_vec, fock_ov_mat, ovvv_vec, Fvv_mat, Lvv_mat);
  auto l_int_end_time = std::chrono::steady_clock::now();
  std::chrono::duration<double> l_int_elapsed = l_int_end_time - l_int_start_time;
  std::cout << "Time to create L intermediates " << l_int_elapsed.count() << " (s)" << std::endl;

  RowTensor4d Woooo(nocc, nocc, nocc, nocc), Wvoov(nvirt, nocc, nocc, nvirt);
  RowTensor4d Wvovo(nvirt, nocc, nvirt, nocc),
      Wvvvv(nvirt, nvirt, nvirt, nvirt);
  Woooo.setZero(), Wvoov.setZero(), Wvovo.setZero(), Wvvvv.setZero();

  Eigen::Map<Eigen::VectorXd> Woooo_vec(Woooo.data(),
                                        nocc * nocc * nocc * nocc);
  Eigen::Map<Eigen::VectorXd> Wvoov_vec(Wvoov.data(),
                                        nvirt * nocc * nocc * nvirt);
  Eigen::Map<Eigen::VectorXd> Wvovo_vec(Wvovo.data(),
                                        nvirt * nocc * nvirt * nocc);
  Eigen::Map<Eigen::VectorXd> Wvvvv_vec(Wvvvv.data(),
                                        nvirt * nvirt * nvirt * nvirt);

  auto w_int_start_time = std::chrono::steady_clock::now();
  make_Woooo(t1_mat, t2_vec, oooo_vec, ovoo_vec, ovov_vec, Woooo_vec);
  make_Wvoov(t1_mat, t2_vec, ovoo_vec, ovov_vec, ovvo_vec, ovvv_vec, Wvoov_vec);
  make_Wvovo(t1_mat, t2_vec, ovoo_vec, ovov_vec, oovv_vec, ovvv_vec, Wvovo_vec);
  make_Wvvvv(t1_mat, t2_vec, ovvv_vec, vvvv_vec, Wvvvv_vec);
  auto w_int_end_time = std::chrono::steady_clock::now();
  std::chrono::duration<double> w_int_elapsed = w_int_end_time - w_int_start_time;
  std::cout << "Time to create W intermediates " << w_int_elapsed.count() << " (s)" << std::endl;

  // tau = t2 + np.einsum("ia,jb->ijab", t1, t1)
  shuffle_idx_4d = {0, 2, 1, 3};
  tmp_4d = t2 + t1.contract(t1, contraction_dims_0d).shuffle(shuffle_idx_4d);

  // t2new += lib.einsum("klij,klab->ijab", Woooo, tau)
  contraction_dims_2d = {Eigen::IndexPair<int>(0, 0),
                         Eigen::IndexPair<int>(1, 1)};
  t2_new += Woooo.contract(tmp_4d, contraction_dims_2d);

  // t2new += lib.einsum("abcd,ijcd->ijab", Wvvvv, tau)
  contraction_dims_2d = {Eigen::IndexPair<int>(2, 2),
                         Eigen::IndexPair<int>(3, 3)};
  shuffle_idx_4d = {2, 3, 0, 1};
  t2_new += Wvvvv.contract(tmp_4d, contraction_dims_2d).shuffle(shuffle_idx_4d);

  // tmp = lib.einsum("ac,ijcb->ijab", Lvv, t2)
  contraction_dims_1d = {Eigen::IndexPair<int>(1, 2)};
  shuffle_idx_4d = {1, 2, 0, 3};
  tmp_4d_2 = Lvv.contract(t2, contraction_dims_1d).shuffle(shuffle_idx_4d);

  // t2new += tmp + tmp.transpose(1, 0, 3, 2)
  shuffle_idx_4d = {1, 0, 3, 2};
  t2_new += tmp_4d_2 + tmp_4d_2.shuffle(shuffle_idx_4d);

  // tmp = lib.einsum("ki,kjab->ijab", Loo, t2)
  contraction_dims_1d = {Eigen::IndexPair<int>(0, 0)};
  tmp_4d_2 = Loo.contract(t2, contraction_dims_1d);

  // t2new -= tmp + tmp.transpose(1, 0, 3, 2)
  shuffle_idx_4d = {1, 0, 3, 2};
  t2_new -= tmp_4d_2 + tmp_4d_2.shuffle(shuffle_idx_4d);

  // tmp = 2 * lib.einsum("akic,kjcb->ijab", Wvoov, t2)
  contraction_dims_2d = {Eigen::IndexPair<int>(1, 0),
                         Eigen::IndexPair<int>(3, 2)};
  shuffle_idx_4d = {1, 2, 0, 3};
  tmp_4d_2 =
      2 * Wvoov.contract(t2, contraction_dims_2d).shuffle(shuffle_idx_4d);

  // tmp -= lib.einsum("akci,kjcb->ijab", Wvovo, t2)
  contraction_dims_2d = {Eigen::IndexPair<int>(1, 0),
                         Eigen::IndexPair<int>(2, 2)};
  shuffle_idx_4d = {1, 2, 0, 3};
  tmp_4d_2 -= Wvovo.contract(t2, contraction_dims_2d).shuffle(shuffle_idx_4d);

  // t2new += tmp + tmp.transpose(1, 0, 3, 2)
  shuffle_idx_4d = {1, 0, 3, 2};
  t2_new += tmp_4d_2 + tmp_4d_2.shuffle(shuffle_idx_4d);

  // tmp = lib.einsum("akic,kjbc->ijab", Wvoov, t2)
  contraction_dims_2d = {Eigen::IndexPair<int>(1, 0),
                         Eigen::IndexPair<int>(3, 3)};
  shuffle_idx_4d = {1, 2, 0, 3};
  tmp_4d_2 = Wvoov.contract(t2, contraction_dims_2d).shuffle(shuffle_idx_4d);

  // t2new -= tmp + tmp.transpose(1, 0, 3, 2)
  shuffle_idx_4d = {1, 0, 3, 2};
  t2_new -= tmp_4d_2 + tmp_4d_2.shuffle(shuffle_idx_4d);

  // tmp = lib.einsum("bkci,kjac->ijab", Wvovo, t2)
  contraction_dims_2d = {Eigen::IndexPair<int>(1, 0),
                         Eigen::IndexPair<int>(2, 3)};
  shuffle_idx_4d = {1, 2, 3, 0};
  tmp_4d_2 = Wvovo.contract(t2, contraction_dims_2d).shuffle(shuffle_idx_4d);

  // t2new -= tmp + tmp.transpose(1, 0, 3, 2)
  shuffle_idx_4d = {1, 0, 3, 2};
  t2_new -= tmp_4d_2 + tmp_4d_2.shuffle(shuffle_idx_4d);

  // Divide by the energies
  RowTensor2d Dia(nocc, nvirt);
  Dia.setZero();
#pragma omp parallel for
  for (int i = 0; i < nocc; i++) {
    for (int a = 0; a < nvirt; a++) {
      Dia(i, a) = mo_e_o(i) - mo_e_v(a);
    }
  }

  RowTensor4d Dijab(nocc, nocc, nvirt, nvirt);
  Dijab.setZero();
#pragma omp parallel for
  for (int i = 0; i < nocc; i++) {
    for (int j = 0; j < nocc; j++) {
      for (int a = 0; a < nvirt; a++) {
        for (int b = 0; b < nvirt; b++) {
          Dijab(i, j, a, b) = mo_e_o(i) + mo_e_o(j) - mo_e_v(a) - mo_e_v(b);
        }
      }
    }
  }

  t1_new /= Dia;
  t2_new /= Dijab;
} // end update_amps

} // namespace RCCSD