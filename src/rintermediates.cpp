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

void make_Lvv(Eigen::Ref<RowMatrixXd> t1_mat,
              Eigen::Ref<Eigen::VectorXd> t2_vec,
              Eigen::Ref<RowMatrixXd> fock_ov_mat,
              Eigen::Ref<Eigen::VectorXd> ovvv_vec,
              Eigen::Ref<RowMatrixXd> Fvv_mat,
              Eigen::Ref<RowMatrixXd> Lvv_mat) {
  // Unpack args
  const int nocc = t1_mat.rows(), nvirt = t1_mat.cols();
  Eigen::TensorMap<RowTensor2d> t1(t1_mat.data(), nocc, nvirt);
  Eigen::TensorMap<RowTensor4d> t2(t2_vec.data(), nocc, nocc, nvirt, nvirt);
  Eigen::TensorMap<RowTensor2d> fock_ov(fock_ov_mat.data(), nocc, nvirt);
  Eigen::TensorMap<RowTensor4d> ovvv(ovvv_vec.data(), nocc, nvirt, nvirt, nvirt);
  Eigen::TensorMap<RowTensor2d> Fvv(Fvv_mat.data(), nvirt, nvirt);
  Eigen::TensorMap<RowTensor2d> Lvv(Lvv_mat.data(), nvirt, nvirt);
  std::cout << "Running make_Lvv" << std::endl;

  // Actual Calculation
  // Lac = cc_Fvv(t1, t2, eris) - np.einsum('kc,ka->ac',fov, t1)
  contraction_dims_1d = {Eigen::IndexPair<int>(0,0)};
  shuffle_idx_2d = {1,0};
  Lvv += Fvv - fock_ov.contract(t1, contraction_dims_1d).shuffle(shuffle_idx_2d);
  
  // Lac += 2*np.einsum('kdac,kd->ac', eris_ovvv, t1)
  contraction_dims_2d = {Eigen::IndexPair<int>(0,0), Eigen::IndexPair<int>(1,1)};
  Lvv += 2 * ovvv.contract(t1, contraction_dims_2d);

  // Lac -=   np.einsum('kcad,kd->ac', eris_ovvv, t1)
  contraction_dims_2d = {Eigen::IndexPair<int>(0,0), Eigen::IndexPair<int>(3,1)};
  Lvv -= ovvv.contract(t1, contraction_dims_2d).shuffle(shuffle_idx_2d);
}

void make_Woooo(Eigen::Ref<RowMatrixXd> t1_mat,
                Eigen::Ref<Eigen::VectorXd> t2_vec,
                Eigen::Ref<Eigen::VectorXd> oooo_vec,
                Eigen::Ref<Eigen::VectorXd> ovoo_vec,
                Eigen::Ref<Eigen::VectorXd> ovov_vec,
                Eigen::Ref<Eigen::VectorXd> Woooo_vec) {
  // Unpack args
  const int nocc = t1_mat.rows(), nvirt = t1_mat.cols();
  Eigen::TensorMap<RowTensor2d> t1(t1_mat.data(), nocc, nvirt);
  Eigen::TensorMap<RowTensor4d> t2(t2_vec.data(), nocc, nocc, nvirt, nvirt);
  Eigen::TensorMap<RowTensor4d> oooo(oooo_vec.data(), nocc, nocc, nocc, nocc);
  Eigen::TensorMap<RowTensor4d> ovoo(ovoo_vec.data(), nocc, nvirt, nocc, nocc);
  Eigen::TensorMap<RowTensor4d> ovov(ovov_vec.data(), nocc, nvirt, nocc, nvirt);
  Eigen::TensorMap<RowTensor4d> Woooo(Woooo_vec.data(), nocc, nocc, nocc, nocc);
  std::cout << "Running make_Woooo" << std::endl;

  // Actual calculation
  // Wklij  = lib.einsum('lcki,jc->klij', eris_ovoo, t1)
  contraction_dims_1d = {Eigen::IndexPair<int>(1,1)};
  shuffle_idx_4d = {1, 0, 2, 3};
  Woooo += ovoo.contract(t1, contraction_dims_1d).shuffle(shuffle_idx_4d);

  // Wklij += lib.einsum('kclj,ic->klij', eris_ovoo, t1)
  shuffle_idx_4d = {0, 1, 3, 2};
  Woooo += ovoo.contract(t1, contraction_dims_1d).shuffle(shuffle_idx_4d);

  // Wklij += lib.einsum('kcld,ijcd->klij', eris_ovov, t2)
  contraction_dims_2d = {Eigen::IndexPair<int>(1,2), Eigen::IndexPair<int>(3,3)};
  Woooo += ovov.contract(t2, contraction_dims_2d);
  
  // Wklij += lib.einsum('kcld,ic,jd->klij', eris_ovov, t1, t1)
  contraction_dims_1d = {Eigen::IndexPair<int>(1,1)};
  RowTensor4d tmp = ovov.contract(t1, contraction_dims_1d);
  contraction_dims_1d = {Eigen::IndexPair<int>(2,1)};
  Woooo += tmp.contract(t1, contraction_dims_1d);
  
  // Wklij += np.asarray(eris.oooo).transpose(0,2,1,3)
  shuffle_idx_4d = {0, 2, 1, 3};
  Woooo += oooo.shuffle(shuffle_idx_4d);

}

void make_Wvvvv(Eigen::Ref<RowMatrixXd> t1_mat,
                Eigen::Ref<Eigen::VectorXd> t2_vec,
                Eigen::Ref<Eigen::VectorXd> ovvv_vec,
                Eigen::Ref<Eigen::VectorXd> vvvv_vec,
                Eigen::Ref<Eigen::VectorXd> Wvvvv_vec) {
  // Unpack args
  const int nocc = t1_mat.rows(), nvirt = t1_mat.cols();
  Eigen::TensorMap<RowTensor2d> t1(t1_mat.data(), nocc, nvirt);
  Eigen::TensorMap<RowTensor4d> t2(t2_vec.data(), nocc, nocc, nvirt, nvirt);
  Eigen::TensorMap<RowTensor4d> ovvv(ovvv_vec.data(), nocc, nvirt, nvirt, nvirt);
  Eigen::TensorMap<RowTensor4d> vvvv(vvvv_vec.data(), nvirt, nvirt, nvirt, nvirt);
  Eigen::TensorMap<RowTensor4d> Wvvvv(Wvvvv_vec.data(), nvirt, nvirt, nvirt, nvirt);
  std::cout << "Running make_Wvvvv" << std::endl;

  // Actual calculation
  // Wabcd  = lib.einsum('kdac,kb->abcd', eris_ovvv,-t1)
  contraction_dims_1d = {Eigen::IndexPair<int>(0,0)};
  shuffle_idx_4d = {1, 3, 2, 0};
  Wvvvv -= ovvv.contract(t1, contraction_dims_1d).shuffle(shuffle_idx_4d);
  
  // Wabcd -= lib.einsum('kcbd,ka->abcd', eris_ovvv, t1)
  shuffle_idx_4d = {3, 1, 0, 2};
  Wvvvv -= ovvv.contract(t1, contraction_dims_1d).shuffle(shuffle_idx_4d);

  // Wabcd += np.asarray(_get_vvvv(eris)).transpose(0,2,1,3)
  shuffle_idx_4d = {0, 2, 1, 3};
  Wvvvv += vvvv.shuffle(shuffle_idx_4d);
}

void make_Wvoov(Eigen::Ref<RowMatrixXd> t1_mat,
                Eigen::Ref<Eigen::VectorXd> t2_vec,
                Eigen::Ref<Eigen::VectorXd> ovoo_vec,
                Eigen::Ref<Eigen::VectorXd> ovov_vec,
                Eigen::Ref<Eigen::VectorXd> ovvo_vec,
                Eigen::Ref<Eigen::VectorXd> ovvv_vec,
                Eigen::Ref<Eigen::VectorXd> Wvoov_vec){
  // Unpack args
  const int nocc = t1_mat.rows(), nvirt = t1_mat.cols();
  Eigen::TensorMap<RowTensor2d> t1(t1_mat.data(), nocc, nvirt);
  Eigen::TensorMap<RowTensor4d> t2(t2_vec.data(), nocc, nocc, nvirt, nvirt);
  Eigen::TensorMap<RowTensor4d> ovoo(ovoo_vec.data(), nocc, nvirt, nocc, nocc);
  Eigen::TensorMap<RowTensor4d> ovov(ovov_vec.data(), nocc, nvirt, nocc, nvirt);
  Eigen::TensorMap<RowTensor4d> ovvo(ovvo_vec.data(), nocc, nvirt, nvirt, nocc);
  Eigen::TensorMap<RowTensor4d> ovvv(ovvv_vec.data(), nocc, nvirt, nvirt, nvirt);
  Eigen::TensorMap<RowTensor4d> Wvoov(Wvoov_vec.data(), nvirt, nocc, nocc, nvirt);
  std::cout << "Running Wvoov" << std::endl;

  // Actual calculation
  // Wakic  = lib.einsum('kcad,id->akic', eris_ovvv, t1)
  contraction_dims_1d = {Eigen::IndexPair<int>(3,1)};
  shuffle_idx_4d = {2, 0, 3, 1};
  Wvoov += ovvv.contract(t1, contraction_dims_1d).shuffle(shuffle_idx_4d);

  // Wakic -= lib.einsum('kcli,la->akic', eris_ovoo, t1)
  contraction_dims_1d = {Eigen::IndexPair<int>(2,0)};
  shuffle_idx_4d = {3, 0, 2, 1};
  Wvoov -= ovoo.contract(t1, contraction_dims_1d).shuffle(shuffle_idx_4d);

  // Wakic += np.asarray(eris.ovvo).transpose(2,0,3,1)
  shuffle_idx_4d = {2, 0, 3, 1};
  Wvoov += ovvo.shuffle(shuffle_idx_4d);
  
  // Wakic -= 0.5*lib.einsum('ldkc,ilda->akic', eris_ovov, t2)
  contraction_dims_2d = {Eigen::IndexPair<int>(0,1), Eigen::IndexPair<int>(1,2)};
  shuffle_idx_4d = {3, 0, 2, 1};
  Wvoov -= 0.5 * ovov.contract(t2, contraction_dims_2d).shuffle(shuffle_idx_4d);

  // Wakic -= 0.5*lib.einsum('lckd,ilad->akic', eris_ovov, t2)
  contraction_dims_2d = {Eigen::IndexPair<int>(0,1), Eigen::IndexPair<int>(3,3)};
  shuffle_idx_4d = {3, 1, 2, 0};
  Wvoov -= 0.5 * ovov.contract(t2, contraction_dims_2d).shuffle(shuffle_idx_4d);

  // Wakic -= lib.einsum('ldkc,id,la->akic', eris_ovov, t1, t1)
  contraction_dims_1d = {Eigen::IndexPair<int>(1,1)};
  RowTensor4d tmp = ovov.contract(t1, contraction_dims_1d);
  contraction_dims_1d = {Eigen::IndexPair<int>(0,0)};
  shuffle_idx_4d = {3, 0, 2, 1};
  Wvoov -= tmp.contract(t1, contraction_dims_1d).shuffle(shuffle_idx_4d);
  
  // Wakic += lib.einsum('ldkc,ilad->akic', eris_ovov, t2)
  contraction_dims_2d = {Eigen::IndexPair<int>(0,1), Eigen::IndexPair<int>(1,3)};
  shuffle_idx_4d = {3, 0, 2, 1};
  Wvoov += ovov.contract(t2, contraction_dims_2d).shuffle(shuffle_idx_4d);

}

void make_Wvovo(Eigen::Ref<RowMatrixXd> t1_mat,
                Eigen::Ref<Eigen::VectorXd> t2_vec,
                Eigen::Ref<Eigen::VectorXd> ovoo_vec,
                Eigen::Ref<Eigen::VectorXd> ovov_vec,
                Eigen::Ref<Eigen::VectorXd> oovv_vec,
                Eigen::Ref<Eigen::VectorXd> ovvv_vec,
                Eigen::Ref<Eigen::VectorXd> Wvovo_vec){
  const int nocc = t1_mat.rows(), nvirt = t1_mat.cols();
  Eigen::TensorMap<RowTensor2d> t1(t1_mat.data(), nocc, nvirt);
  Eigen::TensorMap<RowTensor4d> t2(t2_vec.data(), nocc, nocc, nvirt, nvirt);
  Eigen::TensorMap<RowTensor4d> ovoo(ovoo_vec.data(), nocc, nvirt, nocc, nocc);
  Eigen::TensorMap<RowTensor4d> ovov(ovov_vec.data(), nocc, nvirt, nocc, nvirt);
  Eigen::TensorMap<RowTensor4d> oovv(oovv_vec.data(), nocc, nocc, nvirt, nvirt);
  Eigen::TensorMap<RowTensor4d> ovvv(ovvv_vec.data(), nocc, nvirt, nvirt, nvirt);
  Eigen::TensorMap<RowTensor4d> Wvovo(Wvovo_vec.data(), nvirt, nocc, nvirt, nocc);
  std::cout << "Running Wvovo" << std::endl;

  // Wakci  = lib.einsum('kdac,id->akci', eris_ovvv, t1)
  contraction_dims_1d = {Eigen::IndexPair<int>(1,1)};
  shuffle_idx_4d = {1, 0, 2, 3};
  Wvovo += ovvv.contract(t1, contraction_dims_1d).shuffle(shuffle_idx_4d);

  // Wakci -= lib.einsum('lcki,la->akci', eris_ovoo, t1)
  contraction_dims_1d = {Eigen::IndexPair<int>(0,0)};
  shuffle_idx_4d = {3, 1, 0, 2};
  Wvovo -= ovoo.contract(t1, contraction_dims_1d).shuffle(shuffle_idx_4d);

  // Wakci += np.asarray(eris.oovv).transpose(2,0,3,1)
  shuffle_idx_4d = {2, 0, 3, 1};
  Wvovo += oovv.shuffle(shuffle_idx_4d);

  // Wakci -= 0.5*lib.einsum('lckd,ilda->akci', eris_ovov, t2)
  contraction_dims_2d = {Eigen::IndexPair<int>(0,1), Eigen::IndexPair<int>(3,2)};
  shuffle_idx_4d = {3, 1, 0, 2};  
  Wvovo -= 0.5 * ovov.contract(t2, contraction_dims_2d).shuffle(shuffle_idx_4d);

  // Wakci -= lib.einsum('lckd,id,la->akci', eris_ovov, t1, t1)
  contraction_dims_1d = {Eigen::IndexPair<int>(3,1)};
  RowTensor4d tmp = ovov.contract(t1, contraction_dims_1d);
  contraction_dims_1d = {Eigen::IndexPair<int>(0,0)};
  shuffle_idx_4d = {3, 1, 0, 2};
  Wvovo -= tmp.contract(t1, contraction_dims_1d).shuffle(shuffle_idx_4d);

}