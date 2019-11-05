
#include <cmath>
#include <iostream>
#include <random>
#include <algorithm>
#include <omp.h>
#include <numeric>

#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/IterativeLinearSolvers>

#include <src/cals.hpp>

using namespace Algo;
using std::exp;

CALS::CALS(float alpha, float reg_u, float reg_i, int num_threads, int seed)
    : _alpha(alpha), _reg_u(reg_u), _reg_i(reg_i), _num_threads(num_threads)
{

};

void CALS::init_params(float* U, float* I, int n_users, int n_items, int n_factors){
    RecBase::init_params(_U, _I, U, I, n_users, n_items, n_factors);
    _n_users = n_users;
    _n_items = n_items;
}

using namespace std;
float CALS::update(int* indices, int* indptr, float* data, bool item_side){
    // apply same with item side
    float reg = (item_side) ? _reg_i : _reg_u;
    rowMatrix& I = (item_side) ? _U : _I;
    rowMatrix& U = (item_side) ? _I : _U;
    rowMatrix ItI = I.transpose() * I;

    int n_factors = I.cols();
    int n_users = U.rows();

    Eigen::ConjugateGradient<rowMatrix, Eigen::Lower|Eigen::Upper> cg;

    #pragma omp for schedule(dynamic, 4)
    for(int u = 0; u < n_users; ++u){
        auto num_pos_items = indptr[u + 1] - indptr[u];
        if(0 == num_pos_items)
            continue;

        rowMatrix temp(num_pos_items, n_factors);
        rowMatrix temp2(num_pos_items, n_factors);
        colVector y(n_factors, 1);

        temp.setZero();
        temp2.setZero();
        y.setZero();

        for(int idx = indptr[u], t = 0; idx < indptr[u + 1]; ++idx, ++t){
            int i = indices[idx];
            float val = data[idx];
            temp.row(t) = val * I.row(i);
            temp2.row(t) = I.row(i);
            y.noalias() += (I.row(i).transpose() * (1.0 + val * _alpha));
        }

        rowMatrix YtCY = ItI + (temp.transpose() * temp2 * _alpha);
        for(int d = 0; d < n_factors; ++d)
            YtCY(d, d) += reg;
        cg.setMaxIterations(3).compute(YtCY);
        U.row(u).noalias() = cg.solve(y);
    }

   return 0.0;
}