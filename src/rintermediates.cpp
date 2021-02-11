#include <rintermediates.hpp>

void make_Foo(Eigen::Ref<RowMatrixXd> t1_mat,
              Eigen::Ref<Eigen::VectorXd> t2_vec,
              Eigen::Ref<RowMatrixXd> fock_oo_mat,
              Eigen::Ref<Eigen::VectorXd> ovov_vec,
              Eigen::Ref<RowMatrixXd> Foo_mat)
{
  // Unpack args
  const int nocc = t1_mat.rows(), nvirt = t1_mat.cols();
  Eigen::TensorMap<RowTensor2d> t1(t1_mat.data(), nocc, nvirt);
  Eigen::TensorMap<RowTensor4d> t2(t2_vec.data(), nocc, nocc, nvirt, nvirt);
  Eigen::TensorMap<RowTensor2d> fock_oo(fock_oo_mat.data(), nocc, nocc);
  Eigen::TensorMap<RowTensor4d> ovov(ovov_vec.data(), nocc, nvirt, nocc, nvirt);
  Eigen::TensorMap<RowTensor2d> Foo(Foo_mat.data(), nocc, nocc);
  std::cout << "Running make_Foo" << std::endl;

  // Actual Calculation
  // Fki  = 2*lib.einsum('kcld,ilcd->ki', eris_ovov, t2)
  contraction_dims_3d = {Eigen::IndexPair<int>(1, 2), Eigen::IndexPair<int>(2, 1), Eigen::IndexPair<int>(3, 3)};
  Foo += 2 * ovov.contract(t2, contraction_dims_3d);

  // Fki -=   lib.einsum('kdlc,ilcd->ki', eris_ovov, t2)
  contraction_dims_3d = {Eigen::IndexPair<int>(1, 3), Eigen::IndexPair<int>(2, 1), Eigen::IndexPair<int>(3, 2)};
  Foo -= ovov.contract(t2, contraction_dims_3d);

  // Fki += 2*lib.einsum('kcld,ic,ld->ki', eris_ovov, t1, t1)
  contraction_dims_1d = {Eigen::IndexPair<int>(1,1)};
  RowTensor4d tmp = ovov.contract(t1,contraction_dims_1d);
  contraction_dims_2d = {Eigen::IndexPair<int>(1,0), Eigen::IndexPair<int>(2,1)};
  Foo += 2 * tmp.contract(t1,contraction_dims_2d);

  // Fki -=   lib.einsum('kdlc,ic,ld->ki', eris_ovov, t1, t1)
  contraction_dims_1d = {Eigen::IndexPair<int>(3,1)};
  tmp = ovov.contract(t1, contraction_dims_1d);
  contraction_dims_2d = {Eigen::IndexPair<int>(2,0), Eigen::IndexPair<int>(1,1)};
  Foo -= tmp.contract(t1,contraction_dims_2d);

  // Fki += foo
  Foo += fock_oo;
}

void make_Fvv(Eigen::Ref<RowMatrixXd> t1_mat,
              Eigen::Ref<Eigen::VectorXd> t2_vec,
              Eigen::Ref<RowMatrixXd> fock_vv_mat,
              Eigen::Ref<Eigen::VectorXd> ovov_vec,
              Eigen::Ref<RowMatrixXd> Fvv_mat) {

  // Unpack args
  const int nocc = t1_mat.rows(), nvirt = t1_mat.cols();
  Eigen::TensorMap<RowTensor2d> t1(t1_mat.data(), nocc, nvirt);
  Eigen::TensorMap<RowTensor4d> t2(t2_vec.data(), nocc, nocc, nvirt, nvirt);
  Eigen::TensorMap<RowTensor2d> fock_vv(fock_vv_mat.data(), nvirt, nvirt);
  Eigen::TensorMap<RowTensor4d> ovov(ovov_vec.data(), nocc, nvirt, nocc, nvirt);
  Eigen::TensorMap<RowTensor2d> Fvv(Fvv_mat.data(), nvirt, nvirt);
  std::cout << "Running make_Fvv" << std::endl;

  // Actual calculation
  // Fac  =-2*lib.einsum('kcld,klad->ac', eris_ovov, t2)
  contraction_dims_3d = {Eigen::IndexPair<int>(0,0), Eigen::IndexPair<int>(2,1), Eigen::IndexPair<int>(3,3)};
  shuffle_idx_2d = {1,0};
  Fvv -= 2 * ovov.contract(t2, contraction_dims_3d).shuffle(shuffle_idx_2d);

  // Fac +=   lib.einsum('kdlc,klad->ac', eris_ovov, t2)
  contraction_dims_3d = {Eigen::IndexPair<int>(0,0), Eigen::IndexPair<int>(2,1), Eigen::IndexPair<int>(1,3)};
  Fvv += ovov.contract(t2, contraction_dims_3d).shuffle(shuffle_idx_2d);

  // Fac -= 2*lib.einsum('kcld,ka,ld->ac', eris_ovov, t1, t1)
  contraction_dims_1d = {Eigen::IndexPair<int>(0,0)};
  RowTensor4d tmp = ovov.contract(t1, contraction_dims_1d);
  contraction_dims_2d = {Eigen::IndexPair<int>(1,0), Eigen::IndexPair<int>(2,1)};
  Fvv -= 2 * tmp.contract(t1, contraction_dims_2d).shuffle(shuffle_idx_2d);

  // Fac +=   lib.einsum('kdlc,ka,ld->ac', eris_ovov, t1, t1)
  tmp = ovov.contract(t1, contraction_dims_1d);
  contraction_dims_2d = {Eigen::IndexPair<int>(0,1), Eigen::IndexPair<int>(1,0)};
  Fvv += tmp.contract(t1, contraction_dims_2d).shuffle(shuffle_idx_2d);

  // Fac += fvv
  Fvv += fock_vv;
}

void make_Fov(Eigen::Ref<RowMatrixXd> t1_mat,
              Eigen::Ref<Eigen::VectorXd> t2_vec,
              Eigen::Ref<RowMatrixXd> fock_ov_mat,
              Eigen::Ref<Eigen::VectorXd> ovov_vec,
              Eigen::Ref<RowMatrixXd> Fov_mat) {
  // Unpack args
  const int nocc = t1_mat.rows(), nvirt = t1_mat.cols();
  Eigen::TensorMap<RowTensor2d> t1(t1_mat.data(), nocc, nvirt);
  Eigen::TensorMap<RowTensor4d> t2(t2_vec.data(), nocc, nocc, nvirt, nvirt);
  Eigen::TensorMap<RowTensor2d> fock_ov(fock_ov_mat.data(), nocc, nvirt);
  Eigen::TensorMap<RowTensor4d> ovov(ovov_vec.data(), nocc, nvirt, nocc, nvirt);
  Eigen::TensorMap<RowTensor2d> Fov(Fov_mat.data(), nocc, nvirt);
  std::cout << "Running make_Fov" << std::endl;

  // Actual calculation
  // Fkc  = 2*np.einsum('kcld,ld->kc', eris_ovov, t1)
  contraction_dims_2d = {Eigen::IndexPair<int>(2,0), Eigen::IndexPair<int>(3,1)};
  Fov += 2 * ovov.contract(t1, contraction_dims_2d);

  // Fkc -=   np.einsum('kdlc,ld->kc', eris_ovov, t1)
  contraction_dims_2d = {Eigen::IndexPair<int>(2,0), Eigen::IndexPair<int>(1,1)};
  Fov -= ovov.contract(t1, contraction_dims_2d);

  // Fkc += fov
  Fov += fock_ov;

}

void make_Loo(Eigen::Ref<RowMatrixXd> t1_mat,
              Eigen::Ref<Eigen::VectorXd> t2_vec,
              Eigen::Ref<RowMatrixXd> fock_ov_mat,
              Eigen::Ref<Eigen::VectorXd> ovoo_vec,
              Eigen::Ref<RowMatrixXd> Foo_mat,
              Eigen::Ref<RowMatrixXd> Loo_mat) {
  // Unpack args
  const int nocc = t1_mat.rows(), nvirt = t1_mat.cols();
  Eigen::TensorMap<RowTensor2d> t1(t1_mat.data(), nocc, nvirt);
  Eigen::TensorMap<RowTensor4d> t2(t2_vec.data(), nocc, nocc, nvirt, nvirt);
  Eigen::TensorMap<RowTensor2d> fock_ov(fock_ov_mat.data(), nocc, nvirt);
  Eigen::TensorMap<RowTensor4d> ovoo(ovoo_vec.data(), nocc, nvirt, nocc, nocc);
  Eigen::TensorMap<RowTensor2d> Foo(Foo_mat.data(), nocc, nocc);
  Eigen::TensorMap<RowTensor2d> Loo(Loo_mat.data(), nocc, nocc);
  std::cout << "Running make_Loo" << std::endl;

  // Actual calculation
  // Lki = cc_Foo(t1, t2, eris) + np.einsum('kc,ic->ki',fov, t1)
  contraction_dims_1d = {Eigen::IndexPair<int>(1,1)};
  Loo += Foo + fock_ov.contract(t1, contraction_dims_1d);

  // Lki += 2*np.einsum('lcki,lc->ki', eris_ovoo, t1)
  contraction_dims_2d = {Eigen::IndexPair<int>(0,0), Eigen::IndexPair<int>(1,1)};
  Loo += 2 * ovoo.contract(t1, contraction_dims_2d);

  // Lki -=   np.einsum('kcli,lc->ki', eris_ovoo, t1)
  contraction_dims_2d = {Eigen::IndexPair<int>(2,0), Eigen::IndexPair<int>(1,1)};
  Loo -= ovoo.contract(t1, contraction_dims_2d);
}