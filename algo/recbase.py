import numpy as np
import itertools


class RecBase(object):
    def __init__(self):
        pass

    def init_params(self, n_users, n_items):
        self.U = np.random.normal(loc=0, scale=0.01, size=(n_users, self.n_factors)).astype(np.float32)
        self.I = np.random.normal(loc=0, scale=0.01, size=(n_items, self.n_factors)).astype(np.float32)
        self.solver.init_params(self.U, self.I)
        self.init = True

    def recommend(self, user_id, user_items, N=10):
        user = self.U[user_id]

        liked = set()
        liked.update(user_items[user_id].indices)
        scores = self.I.dot(user)
        count = N + len(liked)
        if count < len(scores):
            ids = np.argpartition(scores, -count)[-count:]
            best = sorted(zip(ids, scores[ids]), key=lambda x: -x[1])
        else:
            best = sorted(enumerate(scores), key=lambda x: -x[1])
        return list(itertools.islice((rec for rec in best if rec[0] not in liked), N))
