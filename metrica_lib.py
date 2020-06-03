# metrica_lib
# Library containing the scoring functions

import numpy as np
import pandas as pd
from mat_operations_lib import *

def direct_index(hic_mat, window_size=100):
	mat_dim = hic_mat.shape[0]
	direct_ind_arr = [np.nan for i in range(mat_dim)]
	diag = hic_mat.diagonal(0)
	for i in range(window_size, mat_dim - window_size):
		direct_ind_arr[i] = 2*diag[i] - np.sum(hic_mat[i,i-window_size:i+window_size])
	return direct_ind_arr

def contrast_index(hic_mat, ci_window_size = 10):
	mat_dim = hic_mat.shape[0]
	contrast_ind_arr = [np.nan for i in range(mat_dim)]
	for i in range(ci_window_size + 1, mat_dim - ci_window_size - 1):
		upper_left = hic_mat[i-ci_window_size:i, i-ci_window_size:i]
		upper_right = hic_mat[i-ci_window_size:i, i+1:i+ci_window_size+1]
		lower_left = hic_mat[i+1:i+ci_window_size+1, i-ci_window_size:i]
		lower_right = hic_mat[i+1:i+ci_window_size+1, i+1:i+ci_window_size+1]
		score = (np.sum(upper_left) + np.sum(lower_right))/(np.sum(upper_right) + np.sum(lower_left))
		contrast_ind_arr[i] = score
	return contrast_ind_arr

def insulation_score(hic_mat, window_size=100):
	mat_dim = hic_mat.shape[0]
	norm_coeff = window_size*window_size
	ins_score_arr = [np.nan for i in range(mat_dim)]
	for i in range(window_size, mat_dim - window_size):
		submat = hic_mat[i-window_size:i,i+1:i+window_size+1]
		ins_score_arr[i] = np.sum(submat)/norm_coeff
	return np.array(ins_score_arr)    

#Just the sum of all contact numbers (+ sum of log values)
def metrica_1(hic_mat, window_size=100, log=True):
	mat_dim = hic_mat.shape[0]
	met_1_arr = [np.nan for i in range(mat_dim)]
	if log:
		met_1_arr_log = [np.nan for i in range(mat_dim)]
	for i in range(window_size, mat_dim-window_size):
		ran1 = hic_mat[i - window_size : i,i]
		ran2 = hic_mat[i,i:i+window_size]
		val = np.sum(ran1 + ran2)
		met_1_arr[i] = val
		if log:
			val_log = np.sum(np.log(ran1 + 1)) + \
							 np.sum(np.log(ran2 + 1))
			met_1_arr_log[i] = val_log
	if log:
		return {"met1" : met_1_arr,
				"met1_log" : met_1_arr_log}
	return met_1_arr

#Sums of the difference between matrices (+ sum of log odds)
def metrica_2(hic_mat, null_mat, window_size=100):
	diff_mat = hic_mat - null_mat
	met2 = metrica_1(diff_mat, window_size, False)
	
	log_diff_mat = np.log(hic_mat + 1) - np.log(null_mat + 1)
	met2_log = metrica_1(log_diff_mat, window_size, False)
	
	return {"met2" : met2,
			"met2_log" : met2_log}

def metrica_3_4(hic_mat, 
				null_mat, 
				null_mat_inv = np.array([]), 
				mult_mat = np.array([]), 
				norm_pwr_val = None,
				window_size=100):
	
	if null_mat_inv.shape[0] == 0:
		null_mat_inv = remove_na(null_mat_inv)
		null_mat_inv = np.linalg.pinv(null_mat)
	if mult_mat.shape[0] == 0:
		mult_mat = np.dot(hic_mat, null_mat_inv)
	mult_mat -= np.identity(hic_mat.shape[0])
	
	if norm_pwr_val:
		mat_dim = hic_mat.shape[0]
		met_4_arr = [np.nan for i in range(mat_dim)]
		s = hic_mat.shape[0]
		norm_arr = np.array([np.power(np.abs(i) + 1, norm_pwr_val) for i in np.arange(-window_size, window_size)])
		for i in range(window_size, hic_mat.shape[0] - window_size):
#            met4 = np.sum()
			mult_mat_range = mult_mat[i, i - window_size : i + window_size]
			met4 = np.dot(norm_arr, mult_mat_range)
			met_4_arr[i] = met4
		return np.array(met_4_arr), null_mat_inv, mult_mat
	
	met_3_arr = metrica_1(mult_mat, window_size, False)
	return np.array(met_3_arr), null_mat_inv, mult_mat

#Cosine_score
def metrica_5(hic_mat, 
			  reduct_factor = 50,
			  window_size=100,
			  u = np.array([]), 
			  s = np.array([]),
			  absolute = True):

	if u.shape[0] == 0:
		u, s, vh = np.linalg.svd(hic_mat, full_matrices=True)
	u = u[:, :reduct_factor]
	s = s[:reduct_factor]
	
	vec_mat = np.dot(u, np.diag(s))
	corr_mat = np.zeros(hic_mat.shape)
	
	size = hic_mat.shape[0]
	for i in range(size):
		for j in range(size):
			norm_prod = np.linalg.norm(vec_mat[i])*np.linalg.norm(vec_mat[j])
			if norm_prod == 0:
				val = 0
			else:
				dot = np.dot(vec_mat[i], vec_mat[j])
				val = dot/norm_prod            
			corr_mat[i][j] = val
	if absolute:
		corr_mat = np.abs(corr_mat)
	met_5_arr = metrica_1(corr_mat, window_size, False)
	return corr_mat, met_5_arr

#Sum of all values within a square incorporating the given value
def metrica_6(hic_mat, window_size=100, log=True):
	mat_dim = hic_mat.shape[0]
	met_6_arr = [np.nan for i in range(mat_dim)]
	if log:
		met_6_arr_log = [np.nan for i in range(mat_dim)]
	for i in range(window_size, mat_dim - window_size):
		temp_mat = hic_mat[i-window_size:i+window_size, i-window_size:i+window_size]
		met_6_arr[i] = np.sum(temp_mat)
		if log:
			met_6_arr_log[i] = np.sum(np.log(temp_mat + 1))
	if log:
		return {"met6" : met_6_arr,
				"met6_log" : met_6_arr_log}
	else:
		return met_6_arr

def calculate_metrics(hic_mat,
					  null_mat,
					  start_arr,
					  end_arr,
					  window_size=100,
					  null_mat_inv = np.array([]),
					  mult_mat = np.array([]),
					  u = np.array([]),
					  s = np.array([]),
					  norm_pwrs = [],
					  name_add = "",
					  met_5_reduct_factor = 50,
					  df_dict = {}):
	
	df_dict["start"] = start_arr
	df_dict["end"] = end_arr

	print("calculating IS")
	IS = insulation_score(hic_mat, window_size)
	df_dict[f"IS{name_add}"] = IS
	
	print("calculating modified IS")
	IS_norm = insulation_score(hic_mat - null_mat, window_size)
	df_dict[f"IS_modified{name_add}"] = IS_norm

	print("calculating DI")
	DI = direct_index(hic_mat, window_size)
	df_dict[f"DI{name_add}"] = DI
	
	print("calculating modified DI")
	DI_norm = direct_index(hic_mat - null_mat, window_size)
	df_dict[f"DI_modified{name_add}"] = DI_norm
	
	print("calculating CI")
	CI = contrast_index(hic_mat)
	df_dict[f"CI{name_add}"] = CI
	
	print("calculating modified CI")
	CI_norm = contrast_index(hic_mat - null_mat)
	df_dict[f"CI_modified{name_add}"] = CI_norm
	
	print("calculating metrica 1")
	met1 = metrica_1(hic_mat, window_size)
	df_dict[f"met1{name_add}"] = met1["met1"]
	df_dict[f"met1_log{name_add}"] = met1["met1_log"]
	
	print("calculating metrica 2")
	met2 = metrica_2(hic_mat, null_mat, window_size)
	df_dict[f"met2{name_add}"] = met2["met2"]
	df_dict[f"met2_log{name_add}"] = met2["met2_log"]
	
	print("calculating metrica 3")
	met3, null_mat_inv, mult_mat = metrica_3_4(hic_mat, 
											   null_mat, 
											   null_mat_inv, 
											   mult_mat, 
											   norm_pwr_val = None,
											   window_size = window_size)
	df_dict[f"met3{name_add}"] = met3
	
	for i in norm_pwrs:
		print(f"calculating metrica 4 with normalization factor {i}")
		n = pwr_norm_mat(hic_mat.shape[0], i)
		met4, null_mat_inv, mult_mat = metrica_3_4(hic_mat, 
												   null_mat, 
												   null_mat_inv, 
												   mult_mat, 
												   i,
												   window_size)
		df_dict[f"met4_{i}{name_add}"] = met4

	print(f"calculating metrica 5 using first {met_5_reduct_factor} principal components (absolute values)")
	if u.shape[0] == 0:
		print("SVD not found, calculating SVD")
		u, s, vh = np.linalg.svd(hic_mat, full_matrices=True)
		
	corr_mat, met_5_arr = metrica_5(hic_mat, 
									met_5_reduct_factor,
									window_size,
									u, 
									s,
									absolute = True)
	
	IS_met5_abs = insulation_score(corr_mat, window_size)
	met6_met5_abs = metrica_6(corr_mat, window_size, False)
	
	df_dict[f"met5_abs{name_add}"] = met_5_arr
	df_dict[f"IS_met5_abs{name_add}"] = IS_met5_abs
	df_dict[f"met6_met5_abs{name_add}"] = met6_met5_abs

	
	corr_mat, met_5_arr = metrica_5(hic_mat, 
									met_5_reduct_factor,
									window_size,
									u, 
									s,
									absolute = False)
	
	IS_met5 = insulation_score(corr_mat, window_size)
	met6_met5 = metrica_6(corr_mat, window_size, False)
	
	df_dict[f"met5{name_add}"] = met_5_arr
	df_dict[f"IS_met5{name_add}"] = IS_met5
	df_dict[f"met6_met5{name_add}"] = met6_met5
	
	print("calculating SVD of a normalized matrix")
	
	norm_mat = hic_mat - null_mat
	u, s, vh = np.linalg.svd(norm_mat, full_matrices=True)
	
	corr_mat, met_5_arr = metrica_5(norm_mat, 
									met_5_reduct_factor,
									window_size,
									u, 
									s,
									absolute = True)
	
	
	
	IS_met5 = insulation_score(corr_mat, window_size)
	met6_met5 = metrica_6(corr_mat, window_size, False)
	
	df_dict[f"met5_norm_abs{name_add}"] = met_5_arr
	df_dict[f"IS_met5_norm_abs{name_add}"] = IS_met5
	df_dict[f"met6_met5_norm_abs{name_add}"] = met6_met5
	
	
	corr_mat, met_5_arr = metrica_5(norm_mat, 
									met_5_reduct_factor,
									window_size,
									u, 
									s,
									absolute = False)
	
	IS_met5 = insulation_score(corr_mat, window_size)
	met6_met5 = metrica_6(corr_mat, window_size, False)
	
	df_dict[f"met5_norm{name_add}"] = met_5_arr
	df_dict[f"IS_met5_norm{name_add}"] = IS_met5
	df_dict[f"met6_met5_norm{name_add}"] = met6_met5

	
	print("calculating metrica 6")
	met6 = metrica_6(hic_mat, window_size)
	df_dict[f"met6{name_add}"] = met6["met6"]
	df_dict[f"met6_log{name_add}"] = met6["met6_log"]
	
	print("calculating metrica 6 for normalized matrix")
	met6 = metrica_6(hic_mat - null_mat, window_size, False)
	df_dict[f"met6_normalized{name_add}"] = met6
	
#    for k in df_dict.keys():
#        print(k, len(df_dict[k]))
	
	return pd.DataFrame.from_dict(df_dict)
	
def merge_dfs(df_arr):
	return pd.concat(df_arr)