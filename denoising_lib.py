# denoising_lib
# Library for the denoising procedures

import numpy as np
import pandas as pd
from mat_operations_lib import *

def threshold_denoising(matrix, threshold_val=10):
	d = matrix.shape[0]
	print(f"Threshold denoising. Values below {threshold_val} are discarded")
	matrix[matrix < threshold_val] = 0
	disc_num = np.sum(matrix < threshold_val)
	print(f"{disc_num} values out of the total {d*d} have been dropped")
	return matrix

def Fourier_denoising(matrix,
					  threshold=0.05,
					  imag_boost = 1.):
	
	print(f"Fourier denoising. Frequencies below {threshold} are discarded, imaginary parts are multiplied by {imag_boost}")
	dropped_real = 0
	dropped_imag = 0
	d = matrix.shape[0]
	
	fft_mat = np.fft.fft2(matrix)
	for i in range(d):
		for j in range(d):
			fft_mat[i][j] = complex(fft_mat[i][j].real,fft_mat[i][j].imag*imag_boost)
			if np.abs(fft_mat[i][j].real) < threshold:
				fft_mat[i][j] = complex(0,fft_mat[i][j].imag)
				dropped_real += 1
			if np.abs(fft_mat[i][j].imag) < threshold:
				fft_mat[i][j] = complex(fft_mat[i][j].real, 0)
				dropped_imag += 1

	out_mat = np.fft.ifft2(fft_mat).real
	
	print(f"{dropped_real} real and {dropped_imag} imaginary coefficients out of the total {d*d} have been dropped")
	
	return out_mat


def SVD_denoising(matrix,
				  u = np.array([]),
				  s = np.array([]),
				  vh = np.array([]),
				  singval_num = 100):
	
	d = matrix.shape[0]
	print(f"SVD denoising. Using first {singval_num} singular values out of {d}")
	if u.shape[0] == 0:
		print("SVD of the original matrix not found, calculating SVD")
		u, s, vh = np.linalg.svd(matrix, full_matrices=True)
	
	u = u[:, :singval_num]
	s = s[:singval_num]
	vh = vh[:singval_num, :]
	
	return np.dot(u, np.dot(np.diag(s), vh)), u, s, vh


def loess_correction(matrix, null_matrix = np.array([]), polynom_degree = 1, mult_factor = 20):
	if null_matrix.shape[0] == 0:
		null_matrix = summary_null_model(matrix, return_mat = True)
	d = matrix.shape[0]
	s_d = mult_factor*int(np.sqrt(d))
	logmat = np.log(matrix + 1)
	log_nullmat = np.log(null_matrix + 1)
	rng = np.random.default_rng()
	i_indices = rng.choice(d, size=s_d, replace=False).astype(int)
	x_vals = []
	y_vals = []
	for i in i_indices:
		j_indices = rng.choice(d, size=s_d, replace=False).astype(int)
		for j in j_indices:
			x_vals.append(np.abs(i-j))
			y_vals.append(logmat[i][j] - log_nullmat[i][j])
	z = np.polyfit(x_vals, y_vals, polynom_degree)
	print(z)
	p = np.poly1d(z)
#    plt.scatter(x_vals, y_vals)
	xp = np.linspace(0, d)
#    plt.plot(xp, p(xp))
#    plt.show()
	for i in range(d):
		for j in range(d):
			if p(np.abs(i-j)) > 0:
				logmat[i][j] = np.max(logmat[i][j] - p(np.abs(i-j)), 0)
	return np.exp(logmat) - 1

def loess_variance_normalization(matrix,  polynom_degree = 1, _iter=0):
	print(f"LOESS-like variance correction with the polynomial of degree {polynom_degree} fitting, iteration {_iter}")
	d = matrix.shape[0]
	norm_mat = np.log(matrix + 1)
	var_arr = []
	d_vals = np.array(list(range(d-2)))
	for i in d_vals:
		diag = norm_mat.diagonal(i)
		var_arr.append(np.std(diag))
	fit_arr_x = []
	fit_arr_y = []
	for i in range(d-2):
		if var_arr[i] != 0:
			fit_arr_x.append(d_vals[i])
			fit_arr_y.append(var_arr[i])       
	z = np.polyfit(fit_arr_x, fit_arr_y, polynom_degree)
	p = np.poly1d(z)
	xp = np.linspace(0, d-2)
	for i in range(d):
		for j in range(d):
			if p(np.abs(i-j)) > 0:                
				norm_mat[i][j] = np.max(norm_mat[i][j]/p(np.abs(i-j)), 0)
#	plt.semilogy(d_vals, var_arr)
#	plt.semilogy(xp, p(xp))
#	plt.show()
	return np.exp(norm_mat) - 1
