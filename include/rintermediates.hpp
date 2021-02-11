#ifndef RINTERMEDIATES_HPP
#define RINTERMEDIATES_HPP
#include <fricc.hpp>

Eigen::array<Eigen::IndexPair<int>, 1> contraction_dims_1d;
Eigen::array<Eigen::IndexPair<int>, 2> contraction_dims_2d;
Eigen::array<Eigen::IndexPair<int>, 3> contraction_dims_3d;
Eigen::array<int, 2> shuffle_idx_2d;
Eigen::array<int, 4> shuffle_idx_4d;

void make_Foo(Eigen::Ref<RowMatrixXd> t1_mat,
              Eigen::Ref<Eigen::VectorXd> t2_vec,
              Eigen::Ref<RowMatrixXd> fock_oo_mat,
              Eigen::Ref<Eigen::VectorXd> ovov_vec,
              Eigen::Ref<RowMatrixXd> Foo_mat);

void make_Fvv(Eigen::Ref<RowMatrixXd> t1_mat,
              Eigen::Ref<Eigen::VectorXd> t2_vec,
              Eigen::Ref<RowMatrixXd> fock_vv_mat,
              Eigen::Ref<Eigen::VectorXd> ovov_vec,
              Eigen::Ref<RowMatrixXd> Fvv_mat);

void make_Fov(Eigen::Ref<RowMatrixXd> t1_mat,
              Eigen::Ref<Eigen::VectorXd> t2_vec,
              Eigen::Ref<RowMatrixXd> fock_ov_mat,
              Eigen::Ref<Eigen::VectorXd> ovov_vec,
              Eigen::Ref<RowMatrixXd> Fov_mat);

void make_Loo(Eigen::Ref<RowMatrixXd> t1_mat,
              Eigen::Ref<Eigen::VectorXd> t2_vec,
              Eigen::Ref<RowMatrixXd> fock_ov_mat,
              Eigen::Ref<Eigen::VectorXd> ovoo_vec,
              Eigen::Ref<RowMatrixXd> Foo_mat,
              Eigen::Ref<RowMatrixXd> Loo_mat);

void make_Lvv(Eigen::Ref<RowMatrixXd> t1_mat,
              Eigen::Ref<Eigen::VectorXd> t2_vec,
              Eigen::Ref<RowMatrixXd> fock_ov_mat,
              Eigen::Ref<Eigen::VectorXd> ovvv_vec,
              Eigen::Ref<RowMatrixXd> Fvv_mat,
              Eigen::Ref<RowMatrixXd> Lvv_mat);

void make_Woooo(Eigen::Ref<RowMatrixXd> t1_mat,
                Eigen::Ref<Eigen::VectorXd> t2_vec,
                Eigen::Ref<Eigen::VectorXd> oooo_vec,
                Eigen::Ref<Eigen::VectorXd> ovoo_vec,
                Eigen::Ref<Eigen::VectorXd> ovov_vec,
                Eigen::Ref<Eigen::VectorXd> Woooo_vec);

void make_Wvvvv(Eigen::Ref<RowMatrixXd> t1_mat,
                Eigen::Ref<Eigen::VectorXd> t2_vec,
                Eigen::Ref<Eigen::VectorXd> ovvv_vec,
                Eigen::Ref<Eigen::VectorXd> vvvv_vec,
                Eigen::Ref<Eigen::VectorXd> Wvvvv_vec);

void make_Wvoov(Eigen::Ref<RowMatrixXd> t1_mat,
                Eigen::Ref<Eigen::VectorXd> t2_vec,
                Eigen::Ref<Eigen::VectorXd> ovoo_vec,
                Eigen::Ref<Eigen::VectorXd> ovov_vec,
                Eigen::Ref<Eigen::VectorXd> ovvo_vec,
                Eigen::Ref<Eigen::VectorXd> ovvv_vec,
                Eigen::Ref<Eigen::VectorXd> Wvoov_vec);
#endif