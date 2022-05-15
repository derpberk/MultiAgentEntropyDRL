from cProfile import label

import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process.kernels import Matern


kernel = Matern(length_scale=0.5)

X = np.mgrid[0:10:100j, 0:10:100j]

X = np.vstack(X.T)

p = np.asarray([[2,5],[4,5],[6,5],[8,8],[6,6],[4,4]])

k = kernel(p,X)

m = np.ones((100,100))

for el in k:

	m -= el.reshape(100,100)

m = np.clip(m,0,1)

"""
fig = plt.figure()
ax = plt.axes(projection='3d')

d = ax.plot_surface(X[:,0].reshape(100,100), X[:,1].reshape(100,100), m, linewidth=0, cmap='gray_r', zorder=2, alpha=0.8)
ax.set_xlabel('$X_1$')
ax.set_ylabel('$X_2$')
ax.set_zlabel('$\sigma(X)$')

#ax.scatter(p[:len(p)//2,0], p[:len(p)//2,1], 1.1, c='k', marker='o', zorder=1, label = 'Agent 1')
#ax.scatter(p[len(p)//2:,0], p[len(p)//2:,1], 1.1, c='k', marker='^', zorder=1, label = 'Agent 2')
#plt.legend()

fig.colorbar(d, shrink=0.5, aspect=5)
plt.show()
"""
