# mat_operations
# additional library providing some operations with HiC-matrices

import numpy as np
from scipy import linalg

def normalize_mat(hic_mat):
	hic_mat = hic_mat.astype(np.float32)
	size = hic_mat.shape[0]
	norm_coeffs = np.array([np.sum(hic_mat[i]) for i in range(size)])
	for i in range(size):
		for j in range(size):
			if norm_coeffs[i] != 0 and norm_coeffs[j] != 0:
				hic_mat[i][j] = np.sqrt(hic_mat[i][j]*hic_mat[i][j]/(norm_coeffs[i]*norm_coeffs[j]))
			elif norm_coeffs[i] == 0 and norm_coeffs[j] == 0:
				hic_mat[i][j] = 0
			elif norm_coeffs[i] == 0:
				hic_mat[i][j] = hic_mat[i][j]/norm_coeffs[j]
			elif norm_coeffs[j] == 0:
				hic_mat[i][j] = hic_mat[i][j]/norm_coeffs[i]
			else:
				raise ValueError
	return hic_mat


def summary_null_model(hic_mat, return_mat = True):
	mat_dim = hic_mat.shape[0]
	nullval_arr = []
	for i in range(mat_dim):
		diag = hic_mat.diagonal(i)
		nullval_arr.append(np.sum(diag)/len(diag))
	if not return_mat:
		return nullval_arr
	return linalg.toeplitz(nullval_arr)

def pwr_norm_mat(shape, pwr):
	toeplitz_arr = [np.power(i, pwr) for i in range(1, shape+1)]
	return linalg.toeplitz(toeplitz_arr)

def remove_na(mat):
	for i in range(mat.shape[0]):
		for j in range(mat.shape[0]):
			if np.isnan(mat[i][j]):
				mat[i][j] = 0.
	return mat

def null_diagonal(hic_mat):
	for i in range(hic_mat.shape[0]):
		hic_mat[i][i] = 0
	return hic_mat

def log_transform(mat):
	return np.log(mat + 1)