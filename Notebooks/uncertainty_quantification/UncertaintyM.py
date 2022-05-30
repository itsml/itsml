import math
import numpy as np
import random
from sklearn import metrics
from scipy import stats
from statistics import mean
import warnings
random.seed(1)

def uncertainty_ent_bays(probs, likelihoods): # three dimentianl array with d1 as datapoints, (d2) the rows as samples and (d3) the columns as probability for each class
	p = np.array(probs)
	# print("prob\n", probs)
	# print("likelihoods in bays", likelihoods)
	entropy = -p*np.ma.log2(p)
	entropy = entropy.filled(0)
	# print("entropy\n", entropy)

	a = np.sum(entropy, axis=2)
	al = a * likelihoods
	a = np.sum(al, axis=1)

	given_axis = 1
	dim_array = np.ones((1,probs.ndim),int).ravel()
	dim_array[given_axis] = -1
	b_reshaped = likelihoods.reshape(dim_array)
	mult_out = probs*b_reshaped
	p_m = np.sum(mult_out, axis=1)

	# p_m = np.mean(p, axis=1) #* likelihoods

	total = -np.sum(p_m*np.ma.log2(p_m), axis=1)
	total = total.filled(0)
	e = total - a
	return total, e, a

#################################################################################################################################################

def accuracy_rejection2(predictions_list, labels_list, uncertainty_list, log=False): # more accurate for calculating area under the curve
	accuracy_list = [] # list containing all the acc lists of all runs
	reject_list = []

	for predictions, uncertainty, labels in zip(predictions_list, uncertainty_list, labels_list):

		predictions = np.array(predictions)
		uncertainty = np.array(uncertainty)

		correctness_map = []
		for x, y in zip(predictions, labels):
			if x == y:
				correctness_map.append(1)
			else:
				correctness_map.append(0)

		correctness_map = np.array(correctness_map)
		sorted_index = np.argsort(-uncertainty, kind='stable')
		uncertainty = uncertainty[sorted_index]
		correctness_map = correctness_map[sorted_index]

		accuracy = [] # list of all acc for a single run
		rejection = []
		for i in range(len(uncertainty)):
			t_correctness = correctness_map[i:]
			rej = (len(uncertainty) - len(t_correctness)) / len(uncertainty)
			acc = t_correctness.sum() / len(t_correctness)

			accuracy.append(acc)
			rejection.append(rej)
		accuracy_list.append(np.array(accuracy))
		reject_list.append(rejection)

	# print(">>>>>> ", reject_list)

	min_rejection_len = 999999999
	for rejection in reject_list:
		if len(rejection) < min_rejection_len:
			min_rejection_len = len(rejection)

	for i, (rejection, accuracy) in enumerate(zip(reject_list,accuracy_list)):
		reject_list[i] = rejection[:min_rejection_len]
		accuracy_list[i] = accuracy[:min_rejection_len]

	accuracy_list = np.array(accuracy_list)
	reject_list = np.array(reject_list, dtype=float)
		
	with warnings.catch_warnings():
		warnings.simplefilter("ignore", category=RuntimeWarning)

		avg_accuracy = np.nanmean(accuracy_list, axis=0)
		steps = np.nanmean(reject_list, axis=0)
		std_error = np.std(accuracy_list, axis=0) / math.sqrt(len(uncertainty_list))

	return avg_accuracy, avg_accuracy - std_error, avg_accuracy + std_error, 9999 , steps*100

def unc_heat_map(predictions_list, labels_list, epist_list, ale_list, log=False):
	unc = np.array(epist_list)
	heat_all = np.zeros((unc.shape[0], unc.shape[1], unc.shape[1]))
	rej = np.zeros((2, unc.shape[1]))

	run_index = 0
	for predictions, epist, ale, labels in zip(predictions_list, epist_list, ale_list, labels_list):
		
		predictions = np.array(predictions)
		epist = np.array(epist)
		ale = np.array(ale)

		correctness_map = []
		for x, y in zip(predictions, labels):
			if x == y:
				correctness_map.append(1)
			else:
				correctness_map.append(0)

		correctness_map    = np.array(correctness_map)
		sorted_index_epist = np.argsort(-epist, kind='stable')
		sorted_index_ale   = np.argsort(-ale, kind='stable')

		for i in range(len(epist)):

			sorted_index_epist_t = sorted_index_epist[i:]
			
			for j in range(len(ale)):
				sorted_index_ale_t = sorted_index_ale[j:] # filter based on aleatoric uncertainty
				intersection_index = np.intersect1d(sorted_index_ale_t, sorted_index_epist_t) # intersection of ale and epist
				t_correctness = correctness_map[intersection_index]
				acc = t_correctness.sum() / len(t_correctness)
				heat_all[run_index][len(ale)-1-j][i] = acc
				# rej[0][j] =  j / len(ale)  # rejection percentage
				# rej[1][i] = (len(ale) - i) / len(ale)
				rej[0][i] =  epist[sorted_index_epist[i]] # uncertainty value
				rej[1][j] = ale[sorted_index_ale[len(ale) - j - 1]]
		run_index += 1
	heat = np.mean(heat_all, axis=0)
	# rej = rej * 100
	rej = np.round(rej, 5)
	return heat, rej

def order_comparison(uncertainty_list1, uncertainty_list2, log=False):
	tau_list = []
	pvalue_list = []
	for unc1, unc2 in zip(uncertainty_list1, uncertainty_list2):
		unc1 = np.array(unc1)
		unc2 = np.array(unc2)
		sorted_index1 = np.argsort(-unc1, kind='stable')
		sorted_index2 = np.argsort(-unc2, kind='stable')
		# unc1 = unc1[sorted_index1]
		# unc2 = unc2[sorted_index2]
		# tau, p_value = stats.kendalltau(sorted_index1, sorted_index2)
		tau, p_value = stats.kendalltau(unc1, unc2)
		# tau, p_value = stats.spearmanr(sorted_index1, sorted_index2)

		if log:
			print(sorted_index1)
			print(sorted_index2)
			print(unc1)
			print(unc2)
			print(f"{tau} pvalue {p_value}")
			print("------------------------------------")
			exit()
		tau_list.append(tau)
		pvalue_list.append(p_value)
	comp = mean(tau_list)
	comp_p = mean(pvalue_list)
	if log:
		print(f">>>>>>>>>>>>>>>>>>>>>  {comp} pvalue {comp_p}")
	return comp, comp_p

def accuracy_rejection(predictions_list, labels_list, uncertainty_list, unc_value=False, log=False): # 2D inputs for average plot -> D1: runs D2: uncertainty data

	accuracy_list = []
	r_accuracy_list = []
	
	steps = np.array(list(range(90)))
	if unc_value:
		steps = uncertainty_list


	for predictions, uncertainty, labels in zip(predictions_list, uncertainty_list, labels_list):

		predictions = np.array(predictions)
		uncertainty = np.array(uncertainty)

		correctness_map = []
		for x, y in zip(predictions, labels):
			if x == y:
				correctness_map.append(1)
			else:
				correctness_map.append(0)

		# uncertainty, correctness_map = zip(*sorted(zip(uncertainty,correctness_map),reverse=False))

		correctness_map = np.array(correctness_map)
		sorted_index = np.argsort(uncertainty, kind='stable')
		uncertainty = uncertainty[sorted_index]
		correctness_map = correctness_map[sorted_index]

		correctness_map = list(correctness_map)
		uncertainty = list(uncertainty)
		data_len = len(correctness_map)
		accuracy = []

		for step_index, x in enumerate(steps):
			if unc_value:
				rejection_index = step_index
			else:
				rejection_index = int(data_len *(len(steps) - x) / len(steps))
			x_correct = correctness_map[:rejection_index].copy()
			x_unc = uncertainty[:rejection_index].copy()
			if log:
				print(f"----------------------------------------------- rejection_index {rejection_index}")
				for c,u in zip(x_correct, x_unc):
					print(f"correctness_map {c} uncertainty {u}")
				# print(f"rejection_index = {rejection_index}\nx_correct {x_correct} \nunc {x_unc}")
			if rejection_index == 0:
				accuracy.append(np.nan) # random.random()
			else:
				accuracy.append(np.sum(x_correct) / rejection_index)
		accuracy_list.append(accuracy)

		# random test plot
		r_accuracy = []
		
		for step_index, x in enumerate(steps):
			random.shuffle(correctness_map)
			if unc_value:
				r_rejection_index = step_index
			else:
				r_rejection_index = int(data_len *(len(steps) - x) / len(steps))

			r_x_correct = correctness_map[:r_rejection_index].copy()
			if r_rejection_index == 0:
				r_accuracy.append(np.nan)
			else:
				r_accuracy.append(np.sum(r_x_correct) / r_rejection_index)

		r_accuracy_list.append(r_accuracy)

	accuracy_list = np.array(accuracy_list)
	r_accuracy_list = np.array(r_accuracy_list)
		
	# print(accuracy_list)
	with warnings.catch_warnings():
		warnings.simplefilter("ignore", category=RuntimeWarning)

		avg_accuracy = np.nanmean(accuracy_list, axis=0)
		avg_r_accuracy = np.nanmean(r_accuracy_list, axis=0)
		std_error = np.std(accuracy_list, axis=0) / math.sqrt(len(uncertainty_list))


	return avg_accuracy, avg_accuracy - std_error, avg_accuracy + std_error, avg_r_accuracy , steps

def roc(probs_list, predictions_list, labels_list, uncertainty_list, unc_value=False, log=False): # 2D inputs for average plot -> D1: runs D2: uncertainty data

	area_list = []

	for predictions, uncertainty, labels in zip(predictions_list, uncertainty_list, labels_list):

		predictions = np.array(predictions)
		uncertainty = np.array(uncertainty)

		correctness_map = []
		for x, y in zip(predictions, labels):
			if x == y:
				correctness_map.append(0) 
			else:
				correctness_map.append(1)
		correctness_map = np.array(correctness_map)

		# probs = np.array(probs)
		# predictions = np.array(predictions)
		# uncertainty = np.array(uncertainty)
		# labels = np.array(labels)

		# fpr, tpr, thresholds = metrics.roc_curve(correctness_map, uncertainty)
		# area = metrics.auc(tpr, fpr)
		if len(np.unique(correctness_map)) == 1:
			# print(correctness_map)
			# print("Skipping")
			continue
		area = metrics.roc_auc_score(correctness_map, uncertainty)
		area_list.append(area)

	area_list = np.array(area_list)
	AUROC_mean = area_list.mean()
	AUROC_std  = area_list.std()

	return AUROC_mean, AUROC_std * 2

def roc_epist(probs_list, predictions_list, labels_list, uncertainty_list, unc_value=False, log=False): # 2D inputs for average plot -> D1: runs D2: uncertainty data

	area_list = []

	for predictions, uncertainty, labels in zip(predictions_list, uncertainty_list, labels_list):

		predictions = np.array(predictions)
		uncertainty = np.array(uncertainty)

		# correctness_map = []
		# for x, y in zip(predictions, labels):
		# 	if x == y:
		# 		correctness_map.append(0) 
		# 	else:
		# 		correctness_map.append(1)
		correctness_map = np.array(labels)

		if len(np.unique(correctness_map)) == 1:
			continue
		# print("------------------------------------ unc and correctness_map shape")
		# print(correctness_map)
		area = metrics.roc_auc_score(correctness_map, uncertainty)
		area_list.append(area)

	area_list = np.array(area_list)
	AUROC_mean = area_list.mean()
	AUROC_std  = area_list.std()

	return AUROC_mean, AUROC_std * 2

def uncertainty_correlation(predictions_list, labels_list, uncertainty_list, log=False): # more accurate for calculating area under the curve
	corr_list = [] # list containing all the acc lists of all runs

	for predictions, uncertainty, labels in zip(predictions_list, uncertainty_list, labels_list):

		predictions = np.array(predictions)
		uncertainty = np.array(uncertainty)

		correctness_map = []
		for x, y in zip(predictions, labels):
			if x == y:
				correctness_map.append(0) # switching the correctness labels just to get positive corr values
			else:
				correctness_map.append(1)

		correctness_map = np.array(correctness_map)
		sorted_index = np.argsort(-uncertainty, kind='stable')
		uncertainty = uncertainty[sorted_index]
		correctness_map = correctness_map[sorted_index]
		count = np.unique(correctness_map)
		if len(count) == 1:
			continue
		corr = stats.pearsonr(uncertainty, correctness_map)
		if log:
			print(f"correctness_map \n{correctness_map.shape} uncertainty \n{uncertainty.shape} \ncorr {corr}")
		corr_list.append(corr)

	corr_list = np.array(corr_list)
	avg_corr = np.nanmean(corr_list, axis=0)
	return avg_corr

def uncertainty_distribution(predictions_list, labels_list, uncertainty_list, log=False): # more accurate for calculating area under the curve
	corr_list = [] # list containing all the acc lists of all runs
	unc_correct_all = np.array([])
	unc_incorrect_all = np.array([])

	for predictions, uncertainty, labels in zip(predictions_list, uncertainty_list, labels_list):

		predictions = np.array(predictions)
		uncertainty = np.array(uncertainty)

		correctness_map = []
		for x, y in zip(predictions, labels):
			if x == y:
				correctness_map.append(1) 
			else:
				correctness_map.append(0)
		
		# sort based on correctness_map

		correctness_map = np.array(correctness_map)
		sorted_index = np.argsort(-correctness_map, kind='stable')
		uncertainty = uncertainty[sorted_index]
		correctness_map = correctness_map[sorted_index]
		count = np.unique(correctness_map)

		# split correct from incorrect
		split_index = 0 # code to find where to split the correctness array
		for i, v in enumerate(correctness_map):
			if v != 1:
				split_index = i
				break
		corrects      = correctness_map[:split_index]
		unc_correct   = uncertainty[:split_index]
		incorrects    = correctness_map[split_index:]
		unc_incorrect = uncertainty[split_index:]
		unc_correct_all = np.concatenate((unc_correct_all, unc_correct))
		unc_incorrect_all = np.concatenate((unc_incorrect_all, unc_incorrect))
	
	return unc_correct_all, unc_incorrect_all
