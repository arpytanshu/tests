
#%%

import numpy as np
import matplotlib.pyplot as plt
def norm_vec_euclidean(v):  den=np.sqrt(np.sum(v**2));  return v/den
def norm_vec_manhattan(v):  den=np.sum(np.abs(v));      return v/den
def norm_vec_maximum__(v):  den=np.max(np.abs(v));      return v/den



fig, axs = plt.subplots(3, 1, figsize=(8,24))
c = ['r', 'g', 'b', 'c', 'm', 'y', 'k']*100
for ax, norm_fn, name in zip(axs, 
                          [norm_vec_euclidean, norm_vec_manhattan, norm_vec_maximum__],
                          ['euclidean', 'manhattan', 'maximum']):
    for ix in range(250):
        x = np.random.randint(-400, 400)/100;
        y = np.random.randint(-400, 400)/100;
        v = np.array([x, y])
        v_norm = norm_fn(v)
        ax.plot([0,v[0]], [0, v[1]], '--', alpha=0.2, c=c[ix])
        ax.plot([0,v_norm[0]], [0, v_norm[1]], c=c[ix])
        ax.title.set_text(name)
        # remove borders
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)



