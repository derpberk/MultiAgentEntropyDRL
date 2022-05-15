import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process.kernels import Matern


kernel = Matern(length_scale=1)

X = np.mgrid[0:10:100j, 0:10:100j]

X = np.vstack(X.T)

p = np.asarray([[2,5],[4,5],[6,5],[8,8],[6,6],[4,4]])

k = kernel(p,X)

m = np.ones((100,100))

for el in k:

	m -= el.reshape(100,100)

m = np.clip(m,0,1)

""" Classic conditioning """
kernel = Matern(length_scale=1)
sigma = kernel(X,X) - kernel(X,p) @ (np.linalg.inv(kernel(p,p)) @ kernel(X,p).T)
m2 = sigma.diagonal().reshape(100,100)

fig, ax = plt.subplots(1,3)

ax[0].imshow(m, cmap='gray_r', origin='lower', vmin=0)
ax[1].imshow(m2, cmap='gray_r', origin='lower', vmin=0)
ax[2].imshow(m2-m, cmap='gray_r', origin='lower')

plt.show()
