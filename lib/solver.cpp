
#include <cmath>
#include <iostream>
#include <random>
#include <algorithm>

#include <Eigen/Dense>
#include <Eigen/Core>

#include <src/solver.hpp>
#include <numeric>

using namespace Algo;

using std::exp;


Solver::Solver()
    : RNG(1541)
{


};

void Solver::init_params(float* U, float* I, int n_users, int n_items, int n_factors){
    _n_factors = n_factors;
    new (&_U) Map<rowMatix>(U, n_users, n_factors);
    new (&_I) Map<rowMatix>(I, n_items, n_factors);
    _rng_neg_items = std::uniform_int_distribution<int>(0, n_items - 1);
}

float Solver::fit_user(int u, int* pos_items, size_t pos_item_len){
    float lr = 0.001;
    float reg = 0.01;
    for(int idx=0; idx < pos_item_len; ++idx)
    {
        auto i_p = pos_items[idx];
        // filter negative items
        int i_n = _rng_neg_items(RNG);
        while(true){
            i_n = _rng_neg_items(RNG);
            if(std::find(pos_items, pos_items + pos_item_len, i_n) == (pos_items + pos_item_len))
                break;
        }

        auto diff = _I.row(i_p) - _I.row(i_n);
        float x_uij = (_U.row(u) * (_I.row(i_p) - _I.row(i_n)).transpose())(0, 0);
        auto sigmoid = exp(x_uij);
        _U.row(u) += lr * (sigmoid * diff - reg * _U.row(u));
        _I.row(i_p) += lr * (sigmoid * _U.row(u) - reg * _I.row(i_p));
        _I.row(i_n) += lr * (sigmoid * (-_U.row(u)) - reg * _I.row(i_n));
    }
    return 0.0;
}