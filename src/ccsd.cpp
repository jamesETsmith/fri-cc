#include <ccsd.hpp>

namespace CCSD
{
    void update_amps(Eigen::Ref<RowMatrixXd> t1,
                     Eigen::Ref<Eigen::VectorXd> t2_vec,
                     Eigen::Ref<RowMatrixXd> t1_new,
                     Eigen::Ref<Eigen::VectorXd> t2_vec_new,
                     Eigen::Ref<Eigen::VectorXd> oooo_vec,
                     Eigen::Ref<Eigen::VectorXd> ovoo_vec,
                     Eigen::Ref<Eigen::VectorXd> oovv_vec,
                     Eigen::Ref<Eigen::VectorXd> ovvo_vec,
                     Eigen::Ref<Eigen::VectorXd> ovov_vec,
                     Eigen::Ref<Eigen::VectorXd> ovvv_vec,
                     Eigen::Ref<Eigen::VectorXd> vvvv_vec,
                     int nocc, int nvirt)
    {
        // Unpack ERIS
        Eigen::TensorMap<RowTensor4d> oooo(oooo_vec.data(), nocc, nocc, nocc, nocc);
        Eigen::TensorMap<RowTensor4d> ovoo(ovoo_vec.data(), nocc, nvirt, nocc, nocc);

        Eigen::TensorMap<RowTensor4d> oovv(oovv_vec.data(), nocc, nocc, nvirt, nvirt);
        Eigen::TensorMap<RowTensor4d> ovvo(ovvo_vec.data(), nocc, nvirt, nvirt, nocc);
        Eigen::TensorMap<RowTensor4d> ovov(ovov_vec.data(), nocc, nvirt, nocc, nvirt);

        Eigen::TensorMap<RowTensor4d> ovvv(ovvv_vec.data(), nocc, nvirt, nvirt, nvirt);
        Eigen::TensorMap<RowTensor4d> vvvv(oooo_vec.data(), nvirt, nvirt, nvirt, nvirt);
    }

} // namespace CCSD