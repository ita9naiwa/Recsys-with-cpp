#pragma once

#include <random>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Core>
#include "util.hpp"
#include "rec_base.hpp"

using namespace Eigen;

namespace Algo{
    class CALS: public RecBase{
    public:
        CALS(float alpha, float reg_u, float reg_i, int num_threads, int seed);
        ~CALS();
        void init();
        void init_params(float* X, float* y, int n_users, int n_items, int n_factors);
        float update(int* indices, int* indptr, float* data, bool item_side);
    private:
        int _n_factors;
        float _alpha;
        float _lr;
        float _reg_u, _reg_i;
        float _reg;
        int _num_threads;
        int _n_users, _n_items;
        rowMatrix _U, _I;
        colVector _y;
        std::vector<std::mt19937> RNG;
        std::vector<std::uniform_int_distribution<int>> _rng_neg_items;
    };
}