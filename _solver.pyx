# cython: experimental_cpp_class_def=True, language_level=3
# -*- coding: utf-8 -*-

STUFF = "Hi"
import numpy as np
cimport numpy as np

cdef extern from "src/solver.hpp" namespace "Algo":
    cdef cppclass Solver:
        Solver(float, float, float, int, int)
        void init_params(float*, float*, int, int, int)
        float fit_user(int, int*, int)
        float fit(int*, int*, int*, int*, int)
cdef class CSolver:
    cdef Solver *obj

    def __cinit__(self, lr, reg_u, reg_i, num_threads, seed):
        self.obj = new Solver(lr, reg_u, reg_i, num_threads, seed)

    def init_params(self, float[:, :] U, float[:, :] I,):
        n_users = U.shape[0]
        n_factors = U.shape[1]
        n_items = I.shape[0]
        self.obj.init_params(&U[0, 0], &I[0, 0], n_users, 30, n_factors)

    def fit_user(self, int u, np.ndarray[int, ndim=1] pos_items):
        cdef int pos_items_len = pos_items.shape[0]
        return self.obj.fit_user(u, &pos_items[0], pos_items_len)

    def fit(self,
            np.ndarray[int, ndim=1] indices,
            np.ndarray[int, ndim=1] indptr,
            np.ndarray[int, ndim=1] rows,
            np.ndarray[int, ndim=1] cols,
            int nnz):
        return self.obj.fit(&indices[0], &indptr[0], &rows[0], &cols[0], nnz)

    def __dealloc__(self):
        del self.obj
