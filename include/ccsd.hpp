#ifndef CCSD_HPP
#define CCSD_HPP
#include <fricc.hpp>

namespace CCSD
{
    void update_amps(Eigen::Ref<RowMatrixXd>,
                     Eigen::Ref<Eigen::VectorXd>,
                     Eigen::Ref<RowMatrixXd>,
                     Eigen::Ref<Eigen::VectorXd>,
                     Eigen::Ref<Eigen::VectorXd>,
                     Eigen::Ref<Eigen::VectorXd>,
                     Eigen::Ref<Eigen::VectorXd>,
                     Eigen::Ref<Eigen::VectorXd>,
                     Eigen::Ref<Eigen::VectorXd>,
                     Eigen::Ref<Eigen::VectorXd>,
                     Eigen::Ref<Eigen::VectorXd>,
                     int, int);
}

#endif