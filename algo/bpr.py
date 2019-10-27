import numpy as np
import algo.cbpr
from .recbase import RecBase


class BPR(RecBase):
    def __init__(self,
                 n_factors=100,
                 lr=0.02,
                 reg_u=0.025,
                 reg_i=0.025,
                 num_threads=8,
                 seed=1541):

        self.n_factors = n_factors
        self.lr = lr
        self.reg_u = reg_u
        self.reg_i = reg_i
        self.num_threads = num_threads

        if num_threads <= 0:
            num_threads = 1

        self.solver = algo.cbpr.BPR_solver(self.lr, self.reg_u, self.reg_i, self.num_threads, seed)
        self.init = False

    def init_params(self, n_users, n_items):
        self.U = np.random.normal(loc=0, scale=0.1, size=(n_users, self.n_factors)).astype(np.float32)
        self.I = np.random.normal(loc=0, scale=0.1, size=(n_items, self.n_factors)).astype(np.float32)
        self.solver.init_params(self.U, self.I)
        self.init = True

    def fit(self, user_items, num_iters=3):
        user_items = user_items.tocsr()
        n_users, n_items = user_items.shape

        if self.init is False:
            self.init_params(n_users, n_items)

        indptr = user_items.indptr
        indices = user_items.indices
        _coo = user_items.tocoo()
        cols = _coo.col
        rows = _coo.row
        nnz = _coo.nnz
        for iter in range(num_iters):
            self.solver.fit(indices, indptr, rows, cols, nnz)
