
'''
matrix eigEn decomposition & clustering
=======================================
'''
import numpy as np
from scipy.linalg import eig
from sklearn.cluster import OPTICS


MAT = np.random.rand(4,4)



# compute degree matrix
# =====================
D = np.zeros((MAT.shape))
d = MAT.sum(axis=0)
np.fill_diagonal(D, d)

# compute laplacian
# =================
L = D - MAT

# eig-decompose
# =============
out = eig(L)

evals, evecs = out
evals = evals.real;
evecs = evecs.real;

# sort eigenvals
# ====-=========
sorted_ix = np.argsort(evals)[::-1]

n_cols = 3
X = evecs[:, sorted_ix][:, :n_cols]
clustering = OPTICS(min_samples=20).fit(X)
print(len(set(clustering.labels_)))
labels = clustering.labels_
