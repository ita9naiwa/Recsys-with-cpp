#pragma once

#include <Eigen/Dense>
#include <Eigen/Core>
using namespace Eigen;


using rowMatix = Matrix<float, Dynamic, Dynamic, RowMajor>;
using colVector = Matrix<float, Dynamic, 1>;
namespace Algo{
    class Solver{
    public:
        Solver(int num_features, float reg);
        void init_params(float* X, float*  y, int n_rows, int n_cols);
        void init_weight(float *w, int sz);
        void print_param();
        void asobi();
        void solve_using_normal_eq();

    private:
        colVector _weight;
        int _num_features;
        float _reg;
        rowMatix _X;
        colVector _y;
    };
}