#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Core>

#include "solver.hpp"

using Mat = Eigen::Matrix2d;


int main()
{
    auto solver = Solver(10, 0.03);
    solver.print_param();
}
