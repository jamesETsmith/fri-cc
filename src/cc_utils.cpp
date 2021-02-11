#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

using RowMatrixXd = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using RowTensor4Xd = Eigen::Tensor<double, 4, Eigen::RowMajor>;

void add_to_00(Eigen::Ref<RowMatrixXd> mat, double level_shift)
{
    mat(0, 0) += level_shift;
}

void add_to_0000(Eigen::Ref<VectorXd> vec, int size, double level_shift)
{
    Eigen::TensorMap<RowTensor4Xd> ten(vec.data(), size, size, size, size);
    ten(0, 0, 0, 0) += level_shift;
}
