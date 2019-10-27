import numpy as np
import algo.cals
from .recbase import RecBase


class ALS(RecBase):
    def __init__(self,
                 n_factors=100,
                 reg_u=0.025,
                 reg_i=0.025,
                 alpha=1.0,
                 num_threads=8,
                 seed=1541):

        self.n_factors = n_factors
        self.reg_u = reg_u
        self.reg_i = reg_i
        self.num_threads = num_threads
        self.alpha = alpha
        if num_threads <= 0:
            num_threads = 1

        self.solver = algo.cals.ALS_solver(self.alpha, self.reg_u, self.reg_i, self.num_threads, seed)
        self.init = False

    def fit(self, user_items, num_iters=3):

        user_items = user_items.tocsr()
        item_users = user_items.transpose().tocsr()

        n_users, n_items = user_items.shape

        if self.init is False:
            self.init_params(n_users, n_items)

        ui_indptr = np.asarray(user_items.indptr)
        ui_indices = np.asarray(user_items.indices)
        ui_data = np.asarray(user_items.data)

        iu_indptr = np.asarray(item_users.indptr)
        iu_indices = np.asarray(item_users.indices)
        iu_data = np.asarray(item_users.data)

        for iter in range(num_iters):
            self.solver.update(ui_indices, ui_indptr, ui_data, False)
            self.solver.update(iu_indices, iu_indptr, iu_data, True)
