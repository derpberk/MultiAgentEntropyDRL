import matplotlib.pyplot as plt
import matplotlib

def cutiestyle(*args, **kwargs):
	def wrapper(*args, **kwargs):
		with plt.style.context(args.get['style']):
			plt.rcParams.update({'font.family': 'serif'})
			args['func'](*args, **kwargs)
	return wrapper


@cutiestyle(style='ggplot')
def cutieplot(ydata, xdata=None, title=None, xlabel=None, ylabel=None):

	if xdata is None:
		plt.plot(ydata)
	else:
		plt.plot(xdata,ydata)

	plt.title(title)
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)

	plt.show()


@cutiestyle(style=None)
def cutieimshow(images, ncolumns, minmax = None, interpolation = None, xlabel=None, ylabel=None, cmaps=None):

	nrows = len(images) // ncolumns
	fig, axs = plt.subplots(nrows=nrows,ncols=ncolumns)

	minmax = [[None,None]] * len(images) if minmax is None else minmax
	interpolation = [None] * len(images) if interpolation is None else interpolation
	xlabel = [None] * len(images) if xlabel is None else xlabel
	ylabel = [None] * len(images)if ylabel is None else ylabel
	cmaps = [None] * len(images)if cmaps is None else cmaps

	for i, ax in enumerate(axs):
		ax.imshow(images[i], interpolation=interpolation[i], cmap=cmaps[i])
		ax.set_xlabel(xlabel[i])
		ax.set_ylabel(ylabel[i])
	plt.show()


if __name__ == '__main__':

	import numpy as np
	y = np.random.rand(100,3)
	x = np.linspace(0,10,100)

	cutieplot(ydata=y,
	          xdata=x,
	          xlabel='Steps',
	          title='MyImage')

	ims = np.random.rand(3,10,10)

	cutieimshow(ims, 1)


