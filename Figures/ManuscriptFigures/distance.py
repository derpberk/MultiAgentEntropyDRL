import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager

plt.rcParams['font.family'] = 'Times New Roman'

def clipped_linear_function(x, max_x, min_x, max_y, min_y):
	""" Clipped linear function given its max_x, min_x, max_y, min_y """
	return np.clip(min_y + (x - min_x) * (max_y - min_y) / (max_x - min_x), min_y, max_y)


mean_distance = np.linspace(0,50,200)

penalization = clipped_linear_function(mean_distance, 30, 5, 0, -1)

plt.style.use('bmh')

plt.plot(mean_distance, penalization, 'k', lw=3)
plt.xlabel(r'Mean distance to the vehicle j, $\; \frac{1}{N} \sum_{\forall j \neq i} d_{ji}$')
plt.ylabel(r'Distance penalization $c_d$')

plt.plot([5,5], [-1, 0], 'k--', lw=1)
plt.text(6,-0.5, r"$d_{min}$", fontdict={'fontsize': 16})
plt.plot([30,30], [-1, 0], 'k--', lw=1)
plt.text(31,-0.5, r"$d_{max}$", fontdict={'fontsize': 16})


plt.show()
