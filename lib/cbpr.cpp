
#include <cmath>
#include <iostream>
#include <random>
#include <algorithm>
#include <omp.h>
#include <numeric>

#include <Eigen/Dense>
#include <Eigen/Core>

#include <src/cbpr.hpp>

using namespace Algo;

using std::exp;
using rowMatrix = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using colVector = Eigen::Matrix<float, Eigen::Dynamic, 1>;

CBPR::CBPR(float lr, float reg_u, float reg_i, int num_threads, int seed)
    : _lr(lr), _reg_u(reg_u), _reg_i(reg_i), _num_threads(num_threads)
{
    for(int i = 0; i < _num_threads; ++i)
        RNG.push_back(std::mt19937(seed + i));

};

void CBPR::init_params(float* U, float* I, int n_users, int n_items, int n_factors){
    RecBase::init_params(_U, _I, U, I, n_users, n_items, n_factors);
}
float CBPR::fit(int* indices, int* indptr, int* rows, int* cols, int nnz){

    // set random number generator
    for(int i = 0; i < _num_threads; ++i)
        _rng_neg_items.push_back(std::uniform_int_distribution<int>(0, nnz - 1 ));
    //모든 유저를 빠짐없이 순회하기 위해서 셔플 수행
    auto _idx_map = std::vector<int>(nnz);
    std::iota(_idx_map.begin(), _idx_map.end(), 0);
    std::random_shuffle(_idx_map.begin(), _idx_map.end());

    #pragma omp parallel for default(shared) schedule(dynamic, 4)
    for(int _idx = 0; _idx < nnz; ++_idx){
        auto idx = _idx_map[_idx];
        auto u = rows[idx];
        auto i_p = cols[idx];

        int tid = omp_get_thread_num();
        int i_n = cols[_rng_neg_items[tid](RNG[tid])];
        while(std::binary_search(indices + indptr[u], indices + indptr[u+1], i_n)){
            i_n = cols[_rng_neg_items[tid](RNG[tid])];
        }

        auto diff = _I.row(i_p) - _I.row(i_n);
        float x_uij = (_U.row(u) * (_I.row(i_p) - _I.row(i_n)).transpose())(0, 0);
        auto sigmoid = 1 / ( 1 + exp(x_uij));

        #pragma omp critical
        {
            _U.row(u) += _lr * (sigmoid * diff - _reg_u * _U.row(u));
            _I.row(i_p) += _lr * (sigmoid * _U.row(u) - _reg_i * _I.row(i_p));
            _I.row(i_n) += _lr * (sigmoid * (-_U.row(u)) - _reg_i * _I.row(i_n));
        }

        // TODO: Add loss calculation and accumulation
    }

    return 0;

}