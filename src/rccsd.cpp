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
  Eigen::array<Eigen::IndexPair<int>, 1> contraction_dims_1d;
  Eigen::array<Eigen::IndexPair<int>, 2> contraction_dims_2d;
  Eigen::array<Eigen::IndexPair<int>, 3> contraction_dims_3d;
  Eigen::array<int, 2> shuffle_idx_2d;
  Eigen::array<int, 4> shuffle_idx_4d;

  //
  // Unpack input args and map to tensors
  //

  const int nocc = t1_mat.rows(), nvirt = t1_mat.cols();
  TMap2d t1(t1_mat.data(), nocc, nvirt);
  TMap4d t2(t2_vec.data(), nocc, nocc, nvirt, nvirt);
  TMap2d t1_new(t1_mat_new.data(), nocc, nvirt);
  TMap4d t2_new(t2_vec_new.data(), nocc, nocc, nvirt, nvirt);

  TMap2d fock_oo(fock_oo_mat.data(), nocc, nocc);
  TMap2d fock_ov(fock_oo_mat.data(), nocc, nvirt);
  TMap2d fock_vv(fock_oo_mat.data(), nvirt, nvirt);

  TMap4d oooo(oooo_vec.data(), nocc, nocc, nocc, nocc);
  TMap4d ovoo(ovoo_vec.data(), nocc, nvirt, nocc, nocc);

  TMap4d oovv(oovv_vec.data(), nocc, nocc, nvirt, nvirt);
  TMap4d ovvo(ovvo_vec.data(), nocc, nvirt, nvirt, nocc);
  TMap4d ovov(ovov_vec.data(), nocc, nvirt, nocc, nvirt);

  TMap4d ovvv(ovvv_vec.data(), nocc, nvirt, nvirt, nvirt);
  TMap4d vvvv(vvvv_vec.data(), nvirt, nvirt, nvirt, nvirt);

  // Move energy terms to the other side
  RowTensor2d Foo(nocc, nocc);
  RowTensor2d Fov(nocc, nocc);
  RowTensor2d Fvv(nocc, nocc);

  Eigen::Map<RowMatrixXd> Foo_mat(Foo.data(), nocc, nocc);
  Eigen::Map<RowMatrixXd> Fov_mat(Fov.data(), nocc, nvirt);
  Eigen::Map<RowMatrixXd> Fvv_mat(Fvv.data(), nvirt, nvirt);

  make_Foo(t1_mat, t2_vec, fock_oo_mat, ovov_vec, Foo_mat);
  make_Fov(t1_mat, t2_vec, fock_ov_mat, ovov_vec, Fov_mat);
  make_Fvv(t1_mat, t2_vec, fock_vv_mat, ovov_vec, Fvv_mat);

  Eigen::VectorXd mo_e_o = mo_energies.head(nocc);
  Eigen::VectorXd mo_e_v = mo_energies.tail(nvirt);
  Foo_mat.diagonal() = Foo_mat.diagonal() - mo_e_o;
  Fvv_mat.diagonal() = Fvv_mat.diagonal() - mo_e_v;

  // T1 Equation update
  // -2*np.einsum('kc,ka,ic->ia', fov, t1, t1)
  contraction_dims_1d = {Eigen::IndexPair<int>(0, 0)};
  RowTensor2d tmp_2d = 2 * fock_ov.contract(t1, contraction_dims_1d);
  contraction_dims_1d = {Eigen::IndexPair<int>(0, 1)};
  shuffle_idx_2d = {1, 0};
  t1_new -= tmp_2d.contract(t1, contraction_dims_1d).shuffle(shuffle_idx_2d);
}

} // namespace RCCSD