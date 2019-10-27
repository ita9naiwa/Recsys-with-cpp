#pragma once

#include <random>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Core>
#include "util.hpp"



namespace Algo{
    class RecBase{
    public:
        //RecBase() = default;
        ~RecBase(){
            if(true == _factor_init)
            {
                new (&_U) Eigen::Map<rowMatrix>(nullptr, 0, 0);
                new (&_I) Eigen::Map<rowMatrix>(nullptr, 0, 0);
            }
        }
        //void init();
        void init_params(rowMatrix &_U, rowMatrix &_I, float* U, float* I, int n_users, int n_items, int n_factors){
            _n_factors = n_factors;
            new (&_U) Eigen::Map<rowMatrix>(U, n_users, n_factors);
            new (&_I) Eigen::Map<rowMatrix>(I, n_items, n_factors);
            _factor_init = true;
        }
    private:
        int _n_factors;
        bool _factor_init = false;
        rowMatrix _U, _I;
    };
}