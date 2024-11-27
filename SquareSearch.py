import numpy as np

class SquareSearch:
    def __init__(self):
        self.vectors = None

    def add(self, vectors):
        self.vectors = vectors

    def search(self, query_vectors, k):
        # res_d = np.array([])
        # res_i = np.array([])
        res_d = []
        res_i = []
        for query_vector in query_vectors:
            distances = np.sum((self.vectors - query_vector) ** 2, axis=1)
            nearest_indices = np.argsort(distances)[:k]
            # res_d = np.append(res_d, distances[nearest_indices], axis=0)
            # res_i = np.append(res_i, nearest_indices, axis=0)
            res_d.append(distances[nearest_indices])
            res_i.append(nearest_indices)

        return np.vstack(res_d), np.vstack(res_i)
