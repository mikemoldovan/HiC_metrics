#struct_scores
#
#Main module of the package

import cooler
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import random
import pandas as pd
from scipy import linalg
from optparse import OptionParser

from denoising_lib import *
from plots_lib import *
from mat_operations_lib import *
from metrica_lib import *

def main(in_cooler_file, 
		 process_id = "", 
		 normalize = True, 
		 null_diagonal = False, 
		 denoising_alg = "",
		 threshold_denoising_param = 10,
		 loess_iter = 3,
		 svd_denoising_param = 100,
		 fourier_denoising_thr = 0.05,
		 fourier_imag_boost = 1.,
		 pwr_list = [-1., -0.5, 0.5, 1.], 
		 window_size = 100, 
		 met5_reduct_factor = 50, 
		 mat_range_start=None, 
		 mat_range_end=None):

	cooler_obj = cooler.Cooler(in_cooler_file)
	cooler_np = cooler_obj.bins()[:].to_numpy()
	if mat_range_start:
		hic_mat = cooler_obj.matrix(balance=False)[mat_range_start:mat_range_end, mat_range_start:mat_range_end]
		start_arr = cooler_np[:,1][mat_range_start:mat_range_end]
		end_arr = cooler_np[:,2][mat_range_start:mat_range_end]
	else:
		hic_mat = cooler_obj.matrix(balance=False)[:,:]
		start_arr = cooler_np[:,1][:]
		end_arr = cooler_np[:,2][:]

# Plot initial maps

	print("Removing NaNs")
	hic_mat = remove_na(hic_mat)

	if "v" in denoising_alg:
		for i in range(loess_iter):
			hic_mat = loess_variance_normalization(hic_mat, _iter = i)
			print("Removing NaNs")
			hic_mat = remove_na(hic_mat)
		hic_mat = loess_correction(hic_mat)
		print("Removing NaNs")
		hic_mat = remove_na(hic_mat)
	if "l" in denoising_alg:
		hic_mat = loess_correction(hic_mat)
		print("Removing NaNs")
		hic_mat = remove_na(hic_mat)
	if "t" in denoising_alg:
		hic_mat = threshold_denoising(hic_mat, threshold_denoising_param)
	if "f" in denoising_alg:
		hic_mat = Fourier_denoising(hic_mat, fourier_denoising_thr, fourier_imag_boost)
	if "s" in denoising_alg:
		print("Calculating SVD of the original matrix")
		u, s, vh = np.linalg.svd(hic_mat, full_matrices=True)
		savefig_vec(s, f"{process_id}_initial_mat_singular_values.png", semlog = True)
		print(f"figure created: {process_id}_initial_mat_singular_values.png")
		hic_mat, u, s, vh = SVD_denoising(hic_mat, u, s, vh, svd_denoising_param)
	if not denoising_alg:
		print("No denoising algorithm provided")

# Plot maps after denoising

	if normalize:
		print("Normalizing matrix")
		hic_mat = normalize_mat(hic_mat)

	if null_diagonal:
		print("Zeroing out diagonal elements")
		hic_mat = null_diagonal(hic_mat)

	print("Calculating the expected contact matrix")
	hic_mat_m0 = summary_null_model(hic_mat, return_mat = True)

	print("Calculating metricas for the initial matrix")
	df = calculate_metrics(hic_mat, 
						   hic_mat_m0, 
						   start_arr, 
						   end_arr, 
						   window_size, 
						   norm_pwrs = pwr_list, 
						   name_add = process_id + "_init_mat", 
						   met_5_reduct_factor = met5_reduct_factor, 
						   df_dict = {})

	df.to_csv(f'{process_id}_metrics.csv', index = False, header=True)

	print("Calculating correlation matrix")
	corr_mat_hic = np.corrcoef(hic_mat)
#	print(hic_mat[100:110, 100:110])
#	print(corr_mat_hic[100:110, 100:110])

	print("Removing NaNs")
	corr_mat_hic = remove_na(corr_mat_hic)

	print("Calculating the expected correlation matrix")
	corr_mat_hic_m0 = summary_null_model(corr_mat_hic, return_mat = True)

	print("Calculating metricas for the correlation matrix")
	df_corr = calculate_metrics(corr_mat_hic, 
								corr_mat_hic_m0, 
								start_arr, 
								end_arr, 
								window_size, 
								norm_pwrs = pwr_list, 
								name_add = process_id + "_corr_mat", 
								met_5_reduct_factor = met5_reduct_factor, 
								df_dict = {})

	df_corr.to_csv(f'{process_id}_correlation_matrix_metrics.csv', index = False, header=True)


parser = OptionParser()
parser.add_option("-i", "--in_cooler_file", help="""
	Mandatory parameter. The name of the HiC-file in .cool format. 
	Try keeping the matrices smaller than 10k rows""")
parser.add_option("-p", "--process_id", help="ID of the run. Will be added to all generated files. Default: None", default='')
parser.add_option("-n", "--normalize", help="""Perform matrix normalization by dividing each value over the 
	corresponding column and row sum 1 for True, 0 for False. Default 0""", default="0")
parser.add_option("-d", "--null_diagonal", help="Zero-out elements on the main diagonal? 1 for True, 0 for False. Default 0", default="0")
parser.add_option("-a", "--denoising_alg", help="""Denoising algorithm.\n\n\t\t\t\t
	'' for none,\n\n\n\t\t\t\t
	t for threshold denoising,\t\t\t\t\n
	f for Fourier denoising,\n\t\t\t\t
	s for SVD denoising, \n\t\t\t\t
	l for the LOESS linear regression mean correction, \n\t\t\t\t
	v for the additional LOESS-like variance correction. loess_iter parameter specifies the number of iterations. \n\t\t\t\t
	Algorithms can be combined, e.g. tf value will result in both threshold and Fourier denoising applied.
	The parameters of each particular algorithm should be specified. Default ''""", default='')
parser.add_option("-t", "--threshold_denoising_param", help="The value, below which the elements in the matrix will be discarded. Default 10", default="10")
parser.add_option("-l", "--loess_iter", help="Number of LOESS variance normalization iterations. Default 3", default="3")
parser.add_option("-s", "--svd_denoising_param", help="The number of principal components used in SVD denoising. Default 100", default="100")
parser.add_option("-u", "--fourier_denoising_thr", help="The threshold value of Fourier coefficient modulus. Default 0.05", default="0.05")
parser.add_option("-b", "--fourier_imag_boost", help="""Multiplier of the Fourier coefficients imaginary parts. 
	Larger multiplier will yield heavier representation of off-diagonal elements. Default 1""", default="1.")
parser.add_option("-w", "--pwr_list", help="""The list of values for the power normalizing matrix. 
	Values below 0 will result in heavier weights assigned to the near-diagonal elements and 
	values above 0 will, respectively, result in heavier weights assigned to the distant off-diagonal elements. Default '0.1,0.2,0.3,0.4,0.5'""", default='0.1,0.2,0.3,0.4,0.5')
parser.add_option("-z", "--window_size", help="Size of the windows some metricas are calculated in. Default 100", default="100")
parser.add_option("-5", "--met5_reduct_factor", help="""The number of principal components used in the 
	SVD reduction-correlation assessement. Should not be larger than svd_denoising_param. Default 50""", default="50")
parser.add_option("-1", "--mat_range_start", help="Start of the specified range. If None, the whole matrix is used. Default None", default="")
parser.add_option("-2", "--mat_range_end", help="End of the specified range. If None, the whole matrix is used. Default None", default="")
opt, args = parser.parse_args() 

pwr_list = map(eval, opt.pwr_list.split(','))
if not opt.mat_range_start:
	mat_range_start = None
	mat_range_end = None
else:
	mat_range_start = eval(opt.mat_range_start)
	mat_range_end = eval(opt.mat_range_end)

main(in_cooler_file = opt.in_cooler_file, 
	 process_id = opt.process_id, 
	 normalize = eval(opt.normalize), 
	 null_diagonal = eval(opt.null_diagonal), 
	 denoising_alg = opt.denoising_alg,
	 threshold_denoising_param = eval(opt.threshold_denoising_param),
	 loess_iter = eval(opt.loess_iter),
	 svd_denoising_param = eval(opt.svd_denoising_param),
	 fourier_denoising_thr = eval(opt.fourier_denoising_thr),
	 fourier_imag_boost = eval(opt.fourier_imag_boost),
	 pwr_list = pwr_list, 
	 window_size = eval(opt.window_size), 
	 met5_reduct_factor = eval(opt.met5_reduct_factor), 
	 mat_range_start = mat_range_start, 
	 mat_range_end = mat_range_end)
