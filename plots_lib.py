# plots_lib
# Library for  the data plotting

import matplotlib
import matplotlib.pyplot as plt

def plot_mat(mat):
	fig, ax = plt.subplots()
	im = ax.imshow(mat, cmap="Blues")
	fig.colorbar(im)
	plt.show()

def plot_vec(vec):
	plt.plot(vec)
	plt.show()

def savefig_mat(mat, figname):
	fig, ax = plt.subplots()
	im = ax.imshow(mat, cmap="Blues")
	fig.colorbar(im)
	plt.savefig(figname)

def savefig_vec(vec, figname, semlog=False):
	if semlog:
		plt.semilogy(vec)
	else:
		plt.plot(vec)
	plt.savefig(figname)