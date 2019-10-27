
#include <cmath>
#include <iostream>
#include <random>
#include <algorithm>
#include <omp.h>

#include <Eigen/Dense>
#include <Eigen/Core>

#include <src/solver.hpp>
#include <numeric>

using namespace Algo;

using std::exp;


using std::cout;
using std::endl;

Solver::Solver(float lr, float reg_u, float reg_i, int num_threads, int seed)
    : _lr(lr), _reg_u(reg_u), _reg_i(reg_i), _num_threads(num_threads)
{
    for(int i = 0; i < _num_threads; ++i)
        RNG.push_back(std::mt19937(seed + i));

};
void Solver::init_params(float* U, float* I, int n_users, int n_items, int n_factors){
    _n_factors = n_factors;
    new (&_U) Map<rowMatix>(U, n_users, n_factors);
    new (&_I) Map<rowMatix>(I, n_items, n_factors);
}
float Solver::fit(int* indices, int* indptr, int* rows, int* cols, int nnz){

    // set random number generator
    for(int i = 0; i < _num_threads; ++i)
        _rng_neg_items.push_back(std::uniform_int_distribution<int>(0, nnz));
    //모든 유저를 빠짐없이 순회하기 위해서... 셔플 수행
    auto _idx_map = std::vector<int>(nnz);
    std::iota(_idx_map.begin(), _idx_map.end(), 0);
    std::random_shuffle(_idx_map.begin(), _idx_map.end());

    #pragma omp parallel for default(shared) num_threads(_num_threads)
    for(int _idx = 0; _idx < nnz; ++_idx){
        auto idx = _idx_map[_idx];
        auto u = rows[idx];
        auto i_p = cols[idx];
        int tid = omp_get_thread_num();

        auto i_n = cols[_rng_neg_items[tid](RNG[tid])];
        while(std::binary_search(indices + indptr[u], indices + indptr[u+1], i_n))
            i_n = cols[_rng_neg_items[tid](RNG[tid])];
        auto diff = _I.row(i_p) - _I.row(i_n);
        float x_uij = (_U.row(u) * (_I.row(i_p) - _I.row(i_n)).transpose())(0, 0);
        auto sigmoid = 1 / ( 1 + exp(x_uij));
        #pragma omp critical
        {
            _U.row(u) += _lr * (sigmoid * diff - _reg_u * _U.row(u));
            _I.row(i_p) += _lr * (sigmoid * _U.row(u) - _reg_i * _I.row(i_p));
            _I.row(i_n) += _lr * (sigmoid * (-_U.row(u)) - _reg_i * _I.row(i_n));
        }

    }

    return 0;

}
float Solver::fit_user(int u, int* pos_items, size_t pos_item_len){
    /*
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
    }*/
    return 0.0;
}