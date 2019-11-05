# cython: experimental_cpp_class_def=True, language_level=3
# -*- coding: utf-8 -*-

STUFF = "Hi"
import numpy as np
cimport numpy as np
from libcpp cimport bool

cdef extern from "src/cals.hpp" namespace "Algo":
    cdef cppclass CALS:
        CALS(float, float, float, int, int)
        void init_params(float*, float*, int, int, int)
        void update(int*, int*, float*, bool)
cdef class ALS_solver:
    cdef CALS *obj

    def __cinit__(self, alpha, reg_u, reg_i, num_threads, seed):
        self.obj = new CALS(alpha, reg_u, reg_i, num_threads, seed)

    def init_params(self, float[:, :] U, float[:, :] I,):
        n_users = U.shape[0]
        n_factors = U.shape[1]
        n_items = I.shape[0]
        self.obj.init_params(&U[0, 0], &I[0, 0], n_users, n_items, n_factors)

    def update(self,
               np.ndarray[int, ndim=1] indices,
               np.ndarray[int, ndim=1] indptr,
               np.ndarray[float, ndim=1] data,
               bool user_side):

        return self.obj.update(&indices[0], &indptr[0], &data[0], user_side)

    def __dealloc__(self):
        del self.obj
