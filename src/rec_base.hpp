#pragma once

#include <random>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Core>
#include "util.hpp"


using namespace Eigen;


namespace Algo{
    class RecBase{
    public:
        //RecBase() = default;
        //~RecBase() = default;
        //void init();
        void init_params(rowMatrix &_U, rowMatrix &_I, float* U, float* I, int n_users, int n_items, int n_factors){
            _n_factors = n_factors;
            new (&_U) Map<rowMatrix>(U, n_users, n_factors);
            new (&_I) Map<rowMatrix>(I, n_items, n_factors);
        }
    private:
        int _n_factors;
        rowMatrix _U, _I;
    };
}