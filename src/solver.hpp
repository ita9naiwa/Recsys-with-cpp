#pragma once

#include <random>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Core>
using namespace Eigen;


using rowMatix = Matrix<float, Dynamic, Dynamic, RowMajor>;
using colVector = Matrix<float, Dynamic, 1>;
namespace Algo{
    class Solver{
    public:
        Solver(float lr, float reg_u, float reg_i, int num_threads, int seed);
        void init();
        void init_params(float* X, float* y, int n_users, int n_items, int n_factors);
        float fit_user(int u, int* pos_items, size_t pos_item_len);
        float fit(int* indices, int* indptr, int* rows, int* cols, int nnz);
    private:
        int _n_factors;
        float _lr;
        float _reg_u, _reg_i;
        float _reg;
        int _num_threads;
        rowMatix _U, _I;
        colVector _y;
        std::vector<std::mt19937> RNG;
        std::vector<std::uniform_int_distribution<int>> _rng_neg_items;
    };
}