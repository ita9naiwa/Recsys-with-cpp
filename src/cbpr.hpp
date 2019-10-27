#pragma once

#include <random>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Core>
#include "util.hpp"
#include "rec_base.hpp"

using namespace Eigen;



namespace Algo{
    class CBPR: public RecBase{
    public:
        CBPR(float lr, float reg_u, float reg_i, int num_threads, int seed);
        ~CBPR();
        void init();
        void init_params(float* U, float* I, int n_users, int n_items, int n_factors);
        float fit(int* indices, int* indptr, int* rows, int* cols, int nnz);
    private:
        int _n_factors;
        float _lr;
        float _reg_u, _reg_i;
        float _reg;
        int _num_threads;
        rowMatrix _U, _I;
        std::vector<std::mt19937> RNG;
        std::vector<std::uniform_int_distribution<int>> _rng_neg_items;
    };
}