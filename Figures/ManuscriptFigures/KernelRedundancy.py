from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel
import numpy as np
import matplotlib.pyplot as plt


kernel = RBF(length_scale=2.0, length_scale_bounds=(1e-1, 1e3))

x = np.linspace(-10, 10, 100).reshape(-1,1)
x1 = np.array([[-3]])
x2 = np.array([[3]])

k1 = 1-kernel(x1, x)
k2 = 1-kernel(x2, x)
# stack the first half of k1 and the second half of k2
p1 = k2.flatten()[:50]
p2 = k1.flatten()[50:]
intersection = 1-np.append(p1,p2)

with plt.style.context('bmh'):
    plt.plot(x.flatten(), k1.flatten(), 'r-', label='$k(p_1, X)$')
    plt.plot(np.tile(x1,2).flatten(), [0,1], 'r--')
    plt.text(x1[0]+0.5, 0.5, '$p_1$', fontsize=14)
    plt.plot(x.flatten(), k2.flatten(), 'b-', label='$k(p_2, X)$')
    plt.plot(np.tile(x2, 2).flatten(), [0, 1], 'b--')
    plt.text(x2[0]+0.5, 0.5, '$p_2$', fontsize=14)
    plt.fill_between(x[:50,0], k1[0,:50].T, k2[0,:50].T,  color='r', alpha=0.2)
    plt.fill_between(x[50:,0], k1[0,50:].T, k2[0,50:].T, color='b', alpha=0.2)
    plt.fill_between(x.flatten(), 1, 1-intersection.flatten(), color='m', alpha=0.2, hatch='/')
    plt.ylabel(r'$\sigma(X)$', fontdict={'fontsize': 14})
    plt.xlabel(r'$X$', fontdict={'fontsize': 14})
    plt.legend()
    plt.show()
    #plt.savefig('KernelRedundancy.pdf')


