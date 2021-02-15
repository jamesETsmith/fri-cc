#ifndef RCCSD_HPP
#define RCCSD_HPP
#include <fricc.hpp>
#include <rintermediates.hpp>

namespace RCCSD {
void update_amps(
    Eigen::Ref<RowMatrixXd> t1_mat, Eigen::Ref<Eigen::VectorXd> t2_vec,
    Eigen::Ref<RowMatrixXd> t1_mat_new, Eigen::Ref<Eigen::VectorXd> t2_vec_new,
    Eigen::Ref<RowMatrixXd> fock_oo_mat, Eigen::Ref<RowMatrixXd> fock_ov_mat,
    Eigen::Ref<RowMatrixXd> fock_vv_mat, Eigen::Ref<Eigen::VectorXd> oooo_vec,
    Eigen::Ref<Eigen::VectorXd> ovoo_vec, Eigen::Ref<Eigen::VectorXd> oovv_vec,
    Eigen::Ref<Eigen::VectorXd> ovvo_vec, Eigen::Ref<Eigen::VectorXd> ovov_vec,
    Eigen::Ref<Eigen::VectorXd> ovvv_vec, Eigen::Ref<Eigen::VectorXd> vvvv_vec,
    Eigen::Ref<Eigen::VectorXd> mo_energies);
}

#endif