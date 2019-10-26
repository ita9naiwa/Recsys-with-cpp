#pragma once

#include <random>

#include <Eigen/Dense>
#include <Eigen/Core>
using namespace Eigen;


using rowMatix = Matrix<float, Dynamic, Dynamic, RowMajor>;
using colVector = Matrix<float, Dynamic, 1>;
namespace Algo{
    class Solver{
    public:
        Solver();
        void init_params(float* X, float* y, int n_users, int n_items, int n_factors);
        float fit_user(int u, int* pos_items, size_t pos_item_len);

    private:
        int _n_factors;
        float _reg;
        rowMatix _U, _I;
        colVector _y;
        std::mt19937 RNG;
        std::uniform_int_distribution<int> _rng_neg_items;
    };
}