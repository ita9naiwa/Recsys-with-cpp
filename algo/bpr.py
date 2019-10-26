import numpy as np

import algo.solver
from scipy.sparse import csr_matrix


class BPR():
    def __init__(self, n_factors=100):
        self.solver = algo.solver.CSolver()
        self.n_factors = n_factors
        self.init = False

    def init_params(self, n_users, n_items):
        self.U = np.random.normal(loc=0, scale=0.01, size=(n_users, self.n_factors)).astype(np.float32)
        self.I = np.random.normal(loc=0, scale=0.01, size=(n_items, self.n_factors)).astype(np.float32)
        self.solver.init_params(self.U, self.I)
        self.init = True

    def fit(self, user_items, num_iters=3):
        user_items = user_items.tocsr()
        n_users, n_items = user_items.shape

        if self.init is False:
            self.init_params(n_users, n_items)

        indptr = user_items.indptr
        indices = user_items.indices

        # TODO threadful run...
        for iter in range(num_iters):
            for u in range(n_users):
                if indptr[u] == indptr[u + 1]:
                    continue
                else:
                    user_seens = indices[indptr[u]:indptr[u + 1]]
                    self.solver.fit_user(u, user_seens)
