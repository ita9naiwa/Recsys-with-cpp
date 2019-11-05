# cython: experimental_cpp_class_def=True, language_level=3
# -*- coding: utf-8 -*-

STUFF = "Hi"
import numpy as np
cimport numpy as np

cdef extern from "src/cbpr.hpp" namespace "Algo":
    cdef cppclass CBPR:
        CBPR(float, float, float, int, int)
        void init_params(float*, float*, int, int, int)
        float fit_user(int, int*, int)
        float fit(int*, int*, int*, int*, int)

cdef class BPR_solver:
    cdef CBPR *obj

    def __cinit__(self, lr, reg_u, reg_i, num_threads, seed):
        self.obj = new CBPR(lr, reg_u, reg_i, num_threads, seed)

    def init_params(self, float[:, :] U, float[:, :] I,):
        n_users = U.shape[0]
        n_factors = U.shape[1]
        n_items = I.shape[0]
        self.obj.init_params(&U[0, 0], &I[0, 0], n_users, n_items, n_factors)

    def fit(self,
            np.ndarray[int, ndim=1] indices,
            np.ndarray[int, ndim=1] indptr,
            np.ndarray[int, ndim=1] rows,
            np.ndarray[int, ndim=1] cols,
            int nnz):
        return self.obj.fit(&indices[0], &indptr[0], &rows[0], &cols[0], nnz)

    def __dealloc__(self):
        del self.obj
