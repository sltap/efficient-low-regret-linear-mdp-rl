import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import numpy as np

from numpy.random import default_rng

from scipy.sparse import csc_matrix

from sys import getsizeof

random_seed = 1542973613
rng = default_rng()


class SketchTransform:
    def __init__(self, n, d, eps, delta):
        self.n = n
        self.d = d
        self.eps = eps
        self.delta = delta
        self.r = None
    # End fn __init__

    def transform(self, x):
        pass
    # End fn transform

    def embedding_dim(self):
        return self.r
    # End fn embedding_dim
    
    def __sizeof__(self):
        pass
# End class SketchTransform


class CountSketchTransform(SketchTransform):
    def __init__(self, n, d, eps=1e-3, delta=0.99, r=None):
        super().__init__(n, d, eps, delta)

        if r is None:
            self.r = min(int((d**2 * (np.log(d/eps))**2)/eps**2 * np.log(1./delta)+1),n) # Embedding dimension
        else:
            self.r = r
        # End if

        row_indptr = np.arange(n+1)
        row_indices = np.zeros(shape=(n,))
        data = np.zeros_like(row_indices)
        for i in range(n):
            row_indices[i] = rng.choice(range(self.r))
            data[i] = rng.choice([1,-1])
        # End for

        self.SM = csc_matrix((data, row_indices, row_indptr), shape=(self.r, n))
        self.SM_size = row_indptr.nbytes + row_indices.nbytes + data.nbytes
    # End fn __init__

    def transform(self, x):
        return self.SM @ x
    # End fn transform
    
    def __sizeof__(self):
        return self.SM_size + getsizeof(self.eps)*2 + getsizeof(self.n)*2
    # End fn __sizeof__
# End class CountSketchTransform
