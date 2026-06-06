from __future__ import print_function
import numpy as np
from scipy.optimize import curve_fit, minimize
from sklearn.decomposition import FactorAnalysis
from copy import deepcopy
import warnings
import time


def invertFast(A, d):
	assert(A.shape[0] == d.shape[0])
	assert(d.shape[1] == 1)

	k = A.shape[1]
	A = np.array(A)
	d_vec = np.array(d)
	d_inv = np.array(1 / d_vec[:, 0])

	inv_d_squared = np.dot(np.atleast_2d(d_inv).T, np.atleast_2d(d_inv))
	M = np.diag(d_inv) - inv_d_squared * np.dot(np.dot(A, np.linalg.inv(np.eye(k, k) + np.dot(A.T, (d_inv * A.T).T))), A.T)

	return M


def Estep(Y, A, mus, sigmas, decay_coef):
	assert(len(Y[0]) == len(A))

	N, D = Y.shape
	D, K = A.shape
	assert((sigmas.shape[0] == D) and (sigmas.shape[1] == 1))
	assert((mus.shape[0] == D) and (mus.shape[1] == 1))

	EX = np.zeros([N, D])
	EXZ = np.zeros([N, D, K])
	EX2 = np.zeros([N, D])
	EZ = np.zeros([N, K])
	EZZT = np.zeros([N, K, K])
	entropy = 0

	for i in range(N):
		Y_i = Y[i, :]
		Y_is_zero = np.abs(Y_i) < 1e-6
		dim = K + D

		mu_c, sigma_c, augmentedA_0, augmentedA_plus, augmented_D, sigma_22_inv = calcConditionalDistribution(A, mus, sigmas, np.array([np.abs(Y_i[j]) < 1e-6 for j in range(D)]), Y_i[~Y_is_zero])

		dim = len(sigma_c)
		matrixToInvert = computeMatrixInLastStep(A, np.abs(Y[i, :]) < 1e-6, sigmas, K, sigma_c, decay_coef, sigma_22_inv)

		if (Y_is_zero).sum() < D:
			magical_matrix = 2 * decay_coef * ((augmented_D * matrixToInvert.T).T + augmentedA_0 * (np.eye(K) - augmentedA_plus.T * sigma_22_inv * augmentedA_plus) * (augmentedA_0.T * matrixToInvert))
		else:
			magical_matrix = 2 * decay_coef * ((augmented_D * matrixToInvert.T).T + augmentedA_0 * (augmentedA_0.T * matrixToInvert))
		magical_matrix[:, :K] = 0

		if (Y_is_zero).sum() < D:
			sigma_xz = np.array(sigma_c - augmented_D * np.array(magical_matrix) - (magical_matrix * augmentedA_0) * ((np.eye(K) - augmentedA_plus.T * sigma_22_inv * augmentedA_plus) * augmentedA_0.T))
		else:
			sigma_xz = np.array(sigma_c - augmented_D * np.array(magical_matrix) - (magical_matrix * augmentedA_0) * augmentedA_0.T)

		mu_xz = np.array(np.matrix(np.eye(dim) - magical_matrix) * np.matrix(mu_c))

		dim = len(sigma_c)
		sign, logdet = np.linalg.slogdet(sigma_xz)
		if not sign > 0:
			logdet = -np.inf
		entropy = entropy + dim/2*np.log(2*np.pi) + logdet/2 + dim/2


		EZ[i, :] = mu_xz[:K, 0]
		EX[i, Y_is_zero] = mu_xz[K:, 0]
		EX2[i, Y_is_zero] = mu_xz[K:, 0] ** 2 + np.diag(sigma_xz[K:, K:])
		EZZT[i, :, :] = np.dot(np.atleast_2d(mu_xz[:K, :]), np.atleast_2d(mu_xz[:K, :].transpose())) + sigma_xz[:K, :K]
		EXZ[i, Y_is_zero, :] = np.dot(mu_xz[K:], mu_xz[:K].transpose()) + sigma_xz[K:, :K]

	return EZ, EZZT, EX, EXZ, EX2, entropy


def applyWoodburyIdentity(A_inv, B_inv, C):
	A_inv = np.matrix(A_inv)
	B_inv = np.matrix(B_inv)
	C = np.matrix(C)

	A_inv_C = (A_inv * C)
	M = A_inv - A_inv_C * np.linalg.inv(B_inv + C.T * A_inv_C) * A_inv_C.T
	return M


def computeMatrixInLastStep(A, zero_indices, sigmas, K, sigma_c, decay_coef, sigma_22_inv):
	A_0 = np.matrix(A[zero_indices, :])
	A_plus = np.matrix(A[~zero_indices, :])
	sigmas_0 = sigmas[zero_indices]

	E_xz = sigma_c[K:, :][:, :K]
	E_00_prime_inv = np.matrix(invertFast(A_0, sigmas_0 ** 2 + 1 / (2. * decay_coef)))
	E_plusplus_inv = sigma_22_inv
	E_0plus = A_0 * A_plus.T

	if (E_plusplus_inv.shape[0] == 0) or (E_plusplus_inv.shape[1] == 0):
		inv_matrix = (1 / (2. * decay_coef)) * E_00_prime_inv

	elif (A_0.shape[0] < A_0.shape[1]):
		inv_matrix = np.linalg.inv(2. * decay_coef * (np.linalg.inv(E_00_prime_inv) - E_0plus * E_plusplus_inv * E_0plus.T))

	else:
		b_inv = np.linalg.inv((np.matrix(A_0).T * E_00_prime_inv) * np.matrix(A_0))
		innermost_inverse = applyWoodburyIdentity(-E_plusplus_inv, b_inv, np.matrix(A_plus))
		inv_matrix = (1 / (2. * decay_coef)) * (E_00_prime_inv - (E_00_prime_inv * A_0) * (A_plus.T * innermost_inverse * A_plus) * (A_0.T * E_00_prime_inv))

	dim = len(sigma_c)
	M = np.zeros([dim, dim])
	M[:K, :K] = np.eye(K)
	M[K:, :K] = -2 * decay_coef * np.dot(inv_matrix, E_xz)
	M[K:, K:] = inv_matrix

	return np.array(M)


def Mstep(Y, EZ, EZZT, EX, EXZ, EX2, oldA, old_mus, old_sigmas, old_decay_coef, singleSigma=False):
	assert(len(Y) == len(EZ))
	N, D = Y.shape
	N, K = EZ.shape

	A = np.zeros([D, K])
	mus = np.zeros([D, 1])
	sigmas = np.zeros([D, 1])
	Y_is_zero = np.abs(Y) < 1e-6

	B = np.eye(K + 1)
	for k1 in range(K):
		for k2 in range(K):
			B[k1][k2] = sum(EZZT[:, k1, k2])
		B[K, :K] = EZ.sum(axis=0)
		B[:K, K] = EZ.sum(axis=0)

	B[K, K] = N

	tiled_EZ = np.tile(np.resize(EZ, [N, 1, K]), [1, D, 1])
	tiled_Y = np.tile(np.resize(Y, [N, D, 1]), [1, 1, K])
	tiled_Y_is_zero = np.tile(np.resize(Y_is_zero, [N, D, 1]), [1, 1, K])

	c = np.zeros([K + 1, D])
	c[K, :] += (Y_is_zero * EX + (1 - Y_is_zero) * Y).sum(axis=0)
	c[:K, :] = (tiled_Y_is_zero * EXZ + (1 - tiled_Y_is_zero) * tiled_Y * tiled_EZ).sum(axis=0).transpose()

	solution = np.dot(np.linalg.inv(B), c)
	A = solution[:K, :].transpose()
	mus = np.atleast_2d(solution[K, :]).transpose()

	EXM = np.zeros([N, D])
	EM = np.zeros([N, D])
	EM2 = np.zeros([N, D])

	tiled_mus = np.tile(mus.transpose(), [N, 1])
	tiled_A = np.tile(np.resize(A, [1, D, K]), [N, 1, 1])

	EXM = (tiled_A * EXZ).sum(axis=2) + tiled_mus * EX
	test_sum = (tiled_A * tiled_EZ).sum(axis=2)
	A_product = np.tile(np.reshape(A, [1, D, K]), [K, 1, 1]) * (np.tile(np.reshape(A, [1, D, K]), [K, 1, 1]).T)

	for i in range(N):
		EM[i, :] = (np.dot(A, EZ[i, :].transpose()) + mus.transpose())
		EZZT_tiled = np.tile(np.reshape(EZZT[i, :, :], [K, 1, K]), [1, D, 1])
		ezzt_sum = (EZZT_tiled * A_product).sum(axis=2).sum(axis=0)
		EM2[i, :] = ezzt_sum + 2 * test_sum[i, :] * tiled_mus[i, :] + tiled_mus[i, :] ** 2

	sigmas = (Y_is_zero * (EX2 - 2 * EXM + EM2) + (1 - Y_is_zero) * (Y ** 2 - 2 * Y * EM + EM2)).sum(axis=0)
	sigmas = np.atleast_2d(np.sqrt(sigmas / N)).transpose()

	if singleSigma:
		sigmas = np.mean(sigmas) * np.ones(sigmas.shape)

	decay_coef = minimize(lambda x: decayCoefObjectiveFn(x, Y, EX2), old_decay_coef, jac=True, bounds=[[1e-8, np.inf]])
	decay_coef = decay_coef.x[0]

	return A, mus, sigmas, decay_coef


def calcConditionalDistribution(A, mus, sigmas, zero_indices, observed_values):
	D, K = A.shape
	dim = D + K

	mu_x = np.zeros([dim, 1])
	mu_x[K:dim, :] = mus
	augmentedA = np.matrix(np.zeros([dim, K]))
	augmentedA[:K, :] = np.eye(K)
	augmentedA[K:, :] = A
	mu_x = np.atleast_2d(mu_x)
	observed_values = np.atleast_2d(observed_values)

	if len(observed_values) == 1:
		observed_values = observed_values.transpose()

	mu_diff = np.matrix(np.atleast_2d(observed_values - mus[~zero_indices]))

	assert(mu_diff.shape[0] == len(observed_values) and mu_diff.shape[1] == 1)
	assert(len(zero_indices) == D)
	assert(zero_indices.sum() == D - len(observed_values))

	augmented_zero_indices = np.array([True for a in range(K)] + list(zero_indices))

	augmentedA_0 = augmentedA[augmented_zero_indices, :]
	augmentedA_plus = augmentedA[~augmented_zero_indices, :]
	sigma_11 = augmentedA_0 * augmentedA_0.T
	sigma_11[K:, K:] = sigma_11[K:, K:] + np.diag(sigmas[zero_indices][:, 0] ** 2)
	augmented_D = np.array([0 for i in range(K)] + list(sigmas[zero_indices][:, 0] ** 2))

	if len(observed_values) == 0:
		sigma_x = augmentedA * augmentedA.T
		sigma_x[K:, K:] = sigma_x[K:, K:] + np.diag(sigmas[zero_indices][:, 0] ** 2)
		return mu_x, sigma_x, augmentedA_0, augmentedA_plus, augmented_D, np.array([[]])

	sigma_22_inv = np.matrix(invertFast(A[~zero_indices, :], sigmas[~zero_indices] ** 2))
	mu_0 = mu_x[augmented_zero_indices, :] + augmentedA_0 * (augmentedA_plus.T * (sigma_22_inv * mu_diff))
	sigma_0 = sigma_11 - augmentedA_0 * (augmentedA_plus.T * sigma_22_inv * augmentedA_plus) * augmentedA_0.T

	assert((mu_0.shape[0] == zero_indices.sum() + K) and (mu_0.shape[1] == 1))
	assert(sigma_0.shape[0] == zero_indices.sum() + K and sigma_0.shape[1] == zero_indices.sum() + K)

	return np.array(mu_0), np.array(sigma_0), augmentedA_0, augmentedA_plus, augmented_D, sigma_22_inv


def decayCoefObjectiveFn(x, Y, EX2):
	with warnings.catch_warnings():
		warnings.simplefilter("ignore")

		y_squared = Y ** 2
		Y_is_zero = np.abs(Y) < 1e-6
		exp_Y_squared = np.exp(-x * y_squared)
		log_exp_Y = np.nan_to_num(np.log(1 - exp_Y_squared))
		exp_ratio = np.nan_to_num(exp_Y_squared / (1 - exp_Y_squared))
		obj = sum(sum(Y_is_zero * (-EX2 * x) + (1 - Y_is_zero) * log_exp_Y))
		grad = sum(sum(Y_is_zero * (-EX2) + (1 - Y_is_zero) * y_squared * exp_ratio))

		if (type(obj) is not np.float64) or (type(grad) is not np.float64):
			raise Exception("Unexpected behavior in optimizing decay coefficient lambda. Please contact emmap1@cs.stanford.edu.")
		if type(obj) is np.float64:
			obj = -np.array([obj])
		if type(grad) is np.float64:
			grad = -np.array([grad])

		return obj, grad


def initializeParams(Y, K, singleSigma=False, makePlot=False):

	N, D = Y.shape
	model = FactorAnalysis(n_components=K)
	zeroedY = deepcopy(Y)
	mus = np.zeros([D, 1])

	for j in range(D):
		non_zero_idxs = np.abs(Y[:, j]) > 1e-6
		mus[j] = zeroedY[:, j].mean()
		zeroedY[:, j] = zeroedY[:, j] - mus[j]

	model.fit(zeroedY)

	A = model.components_.transpose()
	sigmas = np.atleast_2d(np.sqrt(model.noise_variance_)).transpose()
	if singleSigma:
		sigmas = np.mean(sigmas) * np.ones(sigmas.shape)

	means = []
	ps = []
	for j in range(D):
		non_zero_idxs = np.abs(Y[:, j]) > 1e-6
		means.append(Y[non_zero_idxs, j].mean())
		ps.append(1 - non_zero_idxs.mean())

	decay_coef, pcov = curve_fit(lambda x, decay_coef: np.exp(-decay_coef * (x ** 2)), means, ps, p0=.05)
	decay_coef = decay_coef[0]

	mse = np.mean(np.abs(ps - np.exp(-decay_coef * (np.array(means) ** 2))))

	if (mse > 0) and makePlot:
		from matplotlib.pyplot import figure, scatter, plot, title, show
		figure()
		scatter(means, ps)
		plot(np.arange(min(means), max(means), .1), np.exp(-decay_coef * (np.arange(min(means), max(means), .1) ** 2)))
		title('Decay Coef is %2.3f; MSE is %2.3f' % (decay_coef, mse))
		show()

	return A, mus, sigmas, decay_coef


def testInputData(Y):
	if (Y - np.array(Y, dtype='int32')).sum() < 1e-6:
		raise Exception('Your input matrix is entirely integers. It is possible but unlikely that this is correct: ZIFA takes as input LOG read counts, not read counts.')

	Y_is_zero = np.abs(Y) < 1e-6
	if (Y_is_zero).sum() == 0:
		raise Exception('Your input matrix contains no zeros. This is possible but highly unlikely in scRNA-seq data. ZIFA takes as input log read counts.')

	if (Y < 0).sum() > 0:
		raise Exception('Your input matrix contains negative values. ZIFA takes as input log read counts and should not contain negative values.')

	zero_fracs = Y_is_zero.mean(axis=0)
	column_is_all_zero = zero_fracs == 1.
	if column_is_all_zero.sum() > 0:
		raise Exception("Your Y matrix has columns which are entirely zero; please filter out these columns and rerun the algorithm.")


_classic_test_input_data = testInputData


def logllh(Y, A, mu, sigmas, lam, N, D, K):
    Y_is_zero = np.array(np.abs(Y) < 1e-6) + 0.0
    Y2 = Y ** 2
    mus = mu
    decay_coef = lam
    EZ, EZZT, EX, EXZ, EX2, entropy = Estep(Y, A, mus, sigmas, decay_coef)
    EZ2 = np.zeros([N, K])
    for i in range(N):
        EZ2[i,:] = np.diag(EZZT[i,:,:])
        
    A2 = np.zeros([K, K, D])
    for j in range(D):
        A2[:,:,j] = A[j,:].reshape([K,1]) @ A[j,:].reshape([1,K])
    const = - N*(K+D)/2 * np.log(2*np.pi)
    Q_Z = - 1/2*np.sum(EZ2) - N/2*np.sum(np.log(sigmas ** 2))
    tmp_zero = - decay_coef*EX2 - (EX2 - 2*EX*mus.reshape([1,D]) - 2*np.sum(A.reshape([1,D,K])*EXZ, axis=2)) / 2 / (sigmas.reshape([1,D])**2)
    tmp_nonzero = np.log(1-np.exp(-decay_coef*Y2)+Y_is_zero) - (Y2 - 2*Y*mus.reshape([1,D]) - 2*Y*(EZ@A.T)) / 2 / (sigmas.reshape([1,D])**2)
    tmp = - (EZZT.reshape([N, K*K]) @ A2.reshape([K*K, D]) + 2*mus.reshape([1,D])*(EZ@A.T) + mus.reshape([1,D])**2) / 2 / (sigmas.reshape([1,D])**2)
    Q = const + Q_Z + np.sum(tmp_zero*Y_is_zero) + np.sum(tmp_nonzero*(1-Y_is_zero)) + np.sum(tmp)
    
    
    return Q, EZ


def _fit_classic_model(Y, K, singleSigma=False, Y_test=None):
	t0 = time.time()
	Y = deepcopy(Y)
	Y_is_zero = np.array(np.abs(Y) < 1e-6) + 0.0
	Y2 = Y ** 2
	N, D = Y.shape

	if D > 2000:
		print('Warning: this dataset has a large number of genes. If ZIFA takes too long to run, try using block_ZIFA.py instead')

	_classic_test_input_data(Y)

	print('Running zero-inflated factor analysis with N = %i, D = %i, K = %i' % (N, D, K))

	A, mus, sigmas, decay_coef = initializeParams(Y, K, singleSigma=singleSigma)
	for i, M in enumerate([A, mus, sigmas, decay_coef]):
		if np.any(np.isnan(np.array(M))) or np.any(np.isinf(np.array(M))):
			raise Exception('Matrix index %i in list has a NaN or infinite element' % i)

	max_iter = 100
	param_change_thresh = 1e-2
	n_iter = 0

	while n_iter < max_iter:
		print(n_iter)

		EZ, EZZT, EX, EXZ, EX2, entropy = Estep(Y, A, mus, sigmas, decay_coef)

		new_A, new_mus, new_sigmas, new_decay_coef = Mstep(Y, EZ, EZZT, EX, EXZ, EX2, A, mus, sigmas, decay_coef, singleSigma=singleSigma)
		for i, M in enumerate([EZ, EZZT, EX, EXZ, EX2, new_A, new_mus, new_sigmas, new_decay_coef]):
			if np.any(np.isnan(np.array(M))) or np.any(np.isinf(np.array(M))):
				raise Exception('Matrix index %i in list has a NaN or infinite element' % i)

		paramsNotChanging = True
		max_param_change = 0

		for new, old in [[new_A, A], [new_mus, mus], [new_sigmas, sigmas], [new_decay_coef, decay_coef]]:
			rel_param_change = np.mean(np.abs(new - old)) / np.mean(np.abs(new))

			if rel_param_change > max_param_change:
				max_param_change = rel_param_change

			if rel_param_change > param_change_thresh:
				paramsNotChanging = False
				break

		A = new_A
		mus = new_mus
		sigmas = new_sigmas
		decay_coef = new_decay_coef

		if paramsNotChanging:
			print('Param change below threshold %2.3e after %i iterations' % (param_change_thresh, n_iter))
			break

		if n_iter >= max_iter:
			print('Maximum number of iterations reached; terminating loop')
		n_iter += 1

	EZ, EZZT, EX, EXZ, EX2, entropy = Estep(Y, A, mus, sigmas, decay_coef)

	result = {}
	result["A"] = A
	result["mus"] = mus
	result["sigmas"] = sigmas
	result["decay_coef"] = decay_coef
	result["latent"] = EZ
	result["run_time"] = time.time() - t0

	Y_is_zero = np.array(np.abs(Y) < 1e-6) + 0.0
	Y2 = Y ** 2
	EZ2 = np.zeros([N, K])
	for i in range(N):
		EZ2[i,:] = np.diag(EZZT[i,:,:])
	A2 = np.zeros([K, K, D])
	for j in range(D):
		A2[:,:,j] = A[j,:].reshape([K,1]) @ A[j,:].reshape([1,K])
	const = - N*(K+D)/2 * np.log(2*np.pi)
	Q_Z = - 1/2*np.sum(EZ2) - N/2*np.sum(np.log(sigmas ** 2))
	tmp_zero = - decay_coef*EX2 - (EX2 - 2*EX*mus.reshape([1,D]) - 2*np.sum(A.reshape([1,D,K])*EXZ, axis=2)) / 2 / (sigmas.reshape([1,D])**2)
	tmp_nonzero = np.log(1-np.exp(-decay_coef*Y2)+Y_is_zero) - (Y2 - 2*Y*mus.reshape([1,D]) - 2*Y*(EZ@A.T)) / 2 / (sigmas.reshape([1,D])**2)
	tmp = - (EZZT.reshape([N, K*K]) @ A2.reshape([K*K, D]) + 2*mus.reshape([1,D])*(EZ@A.T) + mus.reshape([1,D])**2) / 2 / (sigmas.reshape([1,D])**2)
	Q = const + Q_Z + np.sum(tmp_zero*Y_is_zero) + np.sum(tmp_nonzero*(1-Y_is_zero)) + np.sum(tmp)

	result["logllh"] = Q / N - np.mean(np.sum(Y, axis=-1))


	result_test = {}
	if (not Y_test is None):
		N, D = Y_test.shape
		logllh_v, latent = logllh(Y_test, A, mus, sigmas, decay_coef, N, D, K)
		result_test["logllh"] = logllh_v / N - np.mean(np.sum(Y_test, axis=-1))
		result_test["latent"] = latent

	return result, result_test


	

import numpy as np
import random
from copy import deepcopy
import time




def generateIndices(n_blocks, N, D):
    y_indices_to_use = []
    idxs = list(range(D))
    n_in_block = int(1. * D / n_blocks)

    for i in range(N):
        partition = []
        random.shuffle(idxs)
        n_added = 0

        for block in range(n_blocks):
            start = n_in_block * block
            end = start + n_in_block

            if block < n_blocks - 1:
                idxs_in_block = idxs[start:end]
            else:
                idxs_in_block = idxs[start:]

            partition.append(sorted(idxs_in_block))
            n_added += len(idxs_in_block)

        y_indices_to_use.append(partition)

        if i == 0:
            print('Block sizes', [len(a) for a in partition])

        assert(n_added == D)

    return y_indices_to_use


def combineMatrices(y_indices, all_EZs, all_EZZTs, all_EXs, all_EXZs, all_EX2s):
    n_blocks = len(all_EZs)
    D = sum([len(a) for a in y_indices])
    K = all_EZs[0].shape[1]

    combined_EX = np.zeros([D])
    combined_EXZ = np.zeros([D, K])
    combined_EX2 = np.zeros([D])
    combined_EZ = np.zeros([K])
    combined_EZZT = np.zeros([K, K])

    assert(len(all_EZs) == len(all_EZZTs) == len(all_EXs) == len(all_EXZs) == len(all_EX2s) == len(y_indices))

    for block in range(n_blocks):
        block_indices = y_indices[block]
        combined_EX[block_indices] = all_EXs[block]
        combined_EX2[block_indices] = all_EX2s[block]
        combined_EZ = combined_EZ + all_EZs[block]
        combined_EZZT = combined_EZZT + all_EZZTs[block]
        combined_EXZ[block_indices, :] = all_EXZs[block][0, :, :]

    combined_EZ = combined_EZ / n_blocks
    combined_EZZT = combined_EZZT / n_blocks

    return combined_EZ, combined_EZZT, combined_EX, combined_EXZ, combined_EX2


def testInputData(Y):
    if (Y - np.array(Y, dtype='int32')).sum() < 1e-6:
        raise Exception('Your input matrix is entirely integers. It is possible but unlikely that this is correct: ZIFA takes as input LOG read counts, not read counts.')

    Y_is_zero = np.abs(Y) < 1e-6
    if (Y_is_zero).sum() == 0:
        raise Exception('Your input matrix contains no zeros. This is possible but highly unlikely in scRNA-seq data. ZIFA takes as input log read counts.')

    if (Y < 0).sum() > 0:
        raise Exception('Your input matrix contains negative values. ZIFA takes as input log read counts and should not contain negative values.')

    zero_fracs = Y_is_zero.mean(axis=0)
    column_is_all_zero = zero_fracs == 1.

    if column_is_all_zero.sum() > 0:
        print("Warning: Your Y matrix has %i columns which are entirely zero; filtering these out before continuing." % (column_is_all_zero.sum()))
        Y = Y[:, ~column_is_all_zero]
    elif (zero_fracs > .9).sum() > 0:
        print('Warning: your Y matrix contains genes which are frequently zero. If the algorithm fails to converge, try filtering out genes which are zero more than 80 - 90% of the time, or using standard ZIFA.')

    return Y


def runEMAlgorithm(Y, K, singleSigma=False, n_blocks=None):
    Y = testInputData(Y)
    N, D = Y.shape

    if n_blocks is None:
        n_blocks = int(max(1, D / 500))
        print('Number of blocks has been set to %i' % n_blocks)

    print('Running block zero-inflated factor analysis with N = %i, D = %i, K = %i, n_blocks = %i' % (N, D, K, n_blocks))

    y_indices_to_use = generateIndices(n_blocks, N, D)

    np.random.seed(23)
    A, mus, sigmas, decay_coef = initializeParams(Y, K, singleSigma=singleSigma)
    for i, M in enumerate([A, mus, sigmas, decay_coef]):
        if np.any(np.isnan(np.array(M))) or np.any(np.isinf(np.array(M))):
            raise Exception('Matrix index %i in list has a NaN or infinite element' % i)

    max_iter = 20
    param_change_thresh = 1e-2
    n_iter = 0

    EZ = np.zeros([N, K])
    EZZT = np.zeros([N, K, K])
    EX = np.zeros([N, D])
    EXZ = np.zeros([N, D, K])
    EX2 = np.zeros([N, D])

    while n_iter < max_iter:
        for i in range(N):
            block_EZs = []
            block_EZZTs = []
            block_EXs = []
            block_EXZs = []
            block_EX2s = []

            for block in range(n_blocks):
                y_idxs = y_indices_to_use[i][block]
                Y_to_use = Y[i, y_idxs]
                A_to_use = A[y_idxs, :]
                mus_to_use = mus[y_idxs]
                sigmas_to_use = sigmas[y_idxs]

                block_EZ, block_EZZT, block_EX, block_EXZ, block_EX2, _ = Estep(np.array([Y_to_use]), A_to_use, mus_to_use, sigmas_to_use, decay_coef)
                block_EZs.append(block_EZ)
                block_EZZTs.append(block_EZZT)
                block_EXs.append(block_EX)
                block_EXZs.append(block_EXZ)
                block_EX2s.append(block_EX2)

            EZ[i], EZZT[i], EX[i], EXZ[i], EX2[i] = combineMatrices(y_indices_to_use[i], block_EZs, block_EZZTs, block_EXs, block_EXZs, block_EX2s)

        new_A, new_mus, new_sigmas, new_decay_coef = Mstep(Y, EZ, EZZT, EX, EXZ, EX2, A, mus, sigmas, decay_coef, singleSigma=singleSigma)

        try:
            for i, M in enumerate([EZ, EZZT, EX, EXZ, EX2, new_A, new_mus, new_sigmas, new_decay_coef]):
                if np.any(np.isnan(np.array(M))) or np.any(np.isinf(np.array(M))):
                    raise Exception('Matrix index %i in list has a NaN or infinite element' % i)
        except:
            print("Error: algorithm failed to converge. Usual solutions to this problem: filtering out genes which are zero more than 80 - 90% of the time, or using standard ZIFA. Automatically retrying ZIFA when filtering out genes.")
            return None

        paramsNotChanging = True
        max_param_change = 0
        for new, old in [[new_mus, mus], [new_A, A], [new_sigmas, sigmas], [new_decay_coef, decay_coef]]:
            rel_param_change = np.mean(np.abs(new - old)) / np.mean(np.abs(new))
            if rel_param_change > max_param_change:
                max_param_change = rel_param_change
            if rel_param_change > param_change_thresh:
                paramsNotChanging = False

        A = new_A
        mus = new_mus
        sigmas = new_sigmas
        decay_coef = new_decay_coef

        if paramsNotChanging:
            print('Param change below threshold %2.3e after %i iterations' % (param_change_thresh, n_iter))
            break

        if n_iter >= max_iter:
            print('Maximum number of iterations reached; terminating loop')

        n_iter += 1

    return EZ, A, mus, sigmas, decay_coef, EX


def _fit_block_model(Y, K, singleSigma=False, n_blocks=None, p0_thresh=.95):

    t0 = time.time()
    Y = deepcopy(Y)
    assert(p0_thresh >= 0 and p0_thresh <= 1)

    print('Filtering out all genes which are zero in more than %2.1f%% of samples. To change this, change p0_thresh.' % (p0_thresh * 100))

    Y = Y[:, (np.abs(Y) < 1e-6).mean(axis=0) <= p0_thresh]
    results = runEMAlgorithm(Y, K, singleSigma=singleSigma, n_blocks=n_blocks)

    while results is None:
        Y_is_zero = np.abs(Y) < 1e-6
        max_zero_frac = Y_is_zero.mean(axis=0).max()
        new_max_zero_frac = max_zero_frac * .95
        print('Previously, maximum fraction of zeros for a gene was %2.3f; now lowering that to %2.3f and rerunning ZIFA' % (max_zero_frac, new_max_zero_frac))

        Y = Y[:, Y_is_zero.mean(axis=0) < new_max_zero_frac]
        print('After filtering out genes with too many zeros, %i samples and %i genes' % Y.shape)

        results = runEMAlgorithm(Y, K, singleSigma=singleSigma, n_blocks=n_blocks)

    EZ, A, mus, sigmas, decay_coef, EX = results

    result = {}
    result["A"] = A
    result["mus"] = mus
    result["sigmas"] = sigmas
    result["decay_coef"] = decay_coef
    result["latent"] = EZ
    result["run_time"] = time.time() - t0

    return result

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
import os
import time
import math
import warnings
from typing import Any, Dict, Optional
warnings.filterwarnings('ignore')


class VariationalParams(nn.Module):
    def __init__(self, n_data, n_latent):
        super().__init__()
        self.mu = nn.Parameter(torch.randn(n_data, n_latent) * 0.01)
        self.logvar = nn.Parameter(torch.randn(n_data, n_latent) * 0.01)
        
        self.to(self._get_device())
    
    def _get_device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def forward(self, idx=None):
        if idx is not None:
            mu = self.mu[idx]
            sigma = torch.exp(0.5 * self.logvar[idx])
        else:
            mu = self.mu
            sigma = torch.exp(0.5 * self.logvar)
        return mu, sigma


class Encoder(nn.Module):
    def __init__(self, n_input, n_latent, non_linear=True):
        super().__init__()
        self.n_input = n_input
        self.n_latent = n_latent
        self.non_linear = non_linear
        
        if self.non_linear:
            self.fc = nn.Sequential(
                nn.Linear(n_input, 1024),
                nn.ReLU(),
                nn.Linear(1024, n_input),
                nn.ReLU()
            )
            for m in self.fc.modules():
                if isinstance(m, nn.Linear):
                    m.weight.data.normal_(0, 0.01)
                    m.bias.data.fill_(0)
        
        self.fc_mu = nn.Linear(n_input, n_latent)
        self.fc_logvar = nn.Linear(n_input, n_latent)
        
        self.fc_mu.weight.data.normal_(0, 0.01)
        self.fc_mu.bias.data.fill_(0)
        self.fc_logvar.weight.data.normal_(0, 0.01)
        self.fc_logvar.bias.data.fill_(0)
        
        self.to(self._get_device())
    
    def _get_device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def forward(self, y):
        if self.non_linear:
            hidden = self.fc(y)
        else:
            hidden = y
        mu = self.fc_mu(hidden)
        logvar = self.fc_logvar(hidden)
        sigma = torch.exp(0.5 * logvar)
        return mu, sigma


class Decoder(nn.Module):
    def __init__(self, n_input, n_latent, act_nng_exp=True):
        super().__init__()
        self.n_input = n_input
        self.n_latent = n_latent
        self.act_nng_exp = act_nng_exp
        
        self.W_netp = nn.Parameter(torch.Tensor(n_input, n_latent).normal_(0, 0.01))
        self.b_netp = nn.Parameter(torch.Tensor(n_input).normal_(0, 0.01))
        self.logW = nn.Parameter(torch.randn(n_input))  
        self.loglam = nn.Parameter(torch.randn(1))
        
        self.to(self._get_device())
    
    def _get_device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def forward(self, z):
        x_tilde = torch.matmul(z, self.W_netp.t()) + self.b_netp
        
        if self.act_nng_exp:
            W = torch.exp(self.logW)         
            lam = torch.exp(self.loglam)     
        else:
            W = torch.nn.functional.softplus(self.logW)
            lam = torch.nn.functional.softplus(self.loglam)
        
        return x_tilde, W, lam


class ZIFA_Amortized(nn.Module):
    def __init__(self, n_input, n_latent, non_linear=True, act_nng_exp=True, eps=1e-6):
        super().__init__()
        self.encoder = Encoder(n_input, n_latent, non_linear)
        self.decoder = Decoder(n_input, n_latent, act_nng_exp)
        self.n_latent = n_latent
        self.n_input = n_input
        self.eps = eps
        self.method = 'amortized'
        
        self.to(self._get_device())
    
    def _get_device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def _compute_log_prob(self, y, x_tilde, W, lam):
        log_prob = torch.zeros_like(y)
        
        zero_mask = (y < self.eps).float()
        if zero_mask.sum() > 0:
            term1 = -(x_tilde**2) * lam / (1 + 2 * W * lam + self.eps)
            term2 = -0.5 * torch.log(1 + 2 * W * lam + self.eps)
            log_prob_zero = term1 + term2
            log_prob += zero_mask * log_prob_zero
        
        nonzero_mask = (y >= self.eps).float()
        if nonzero_mask.sum() > 0:
            term1 = -(y - x_tilde)**2 / (2 * W + self.eps)
            term2 = -0.5 * math.log(2 * math.pi) - 0.5 * torch.log(W + self.eps)
            term3 = torch.log(1 - torch.exp(-lam * (y**2)) + self.eps)
            log_prob_nonzero = term1 + term2 + term3
            log_prob += nonzero_mask * log_prob_nonzero
        
        return log_prob
    
    def model(self, y):
        pyro.module("decoder", self.decoder)
        batch_size = y.size(0)
        
        with pyro.plate("data", batch_size):
            z_loc = y.new_zeros([batch_size, self.n_latent])
            z_scale = y.new_ones([batch_size, self.n_latent])
            z = pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))
            
            x_tilde, W, lam = self.decoder(z)
            
            log_prob = self._compute_log_prob(y, x_tilde, W, lam)
            pyro.factor("obs", log_prob.sum(-1))
    
    def guide(self, y):
        pyro.module("encoder", self.encoder)
        batch_size = y.size(0)
        
        with pyro.plate("data", batch_size):
            mu_z, sigma_z = self.encoder(y)
            pyro.sample("latent", dist.Normal(mu_z, sigma_z).to_event(1))
    
    def get_latent(self, y=None):
        with torch.no_grad():
            if y is None:
                raise ValueError("y must be provided for amortized inference")
            if y.device != self.device:
                y = y.to(self.device)
            mu_z, _ = self.encoder(y)
        return mu_z
    
    @property
    def device(self):
        return next(self.parameters()).device


class ZIFA_VI(nn.Module):
    def __init__(self, n_input, n_latent, act_nng_exp=True, eps=1e-6):
        super().__init__()
        self.decoder = Decoder(n_input, n_latent, act_nng_exp)
        self.variational_params = None
        self.n_latent = n_latent
        self.n_input = n_input
        self.eps = eps
        self.method = 'vi'
        
        self.to(self._get_device())
    
    def _get_device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def _init_variational_params(self, n_data):
        if self.variational_params is None:
            self.variational_params = VariationalParams(n_data, self.n_latent)
            self.variational_params.to(self.device)
    
    def _compute_log_prob(self, y, x_tilde, W, lam):
        log_prob = torch.zeros_like(y)
        
        zero_mask = (y < self.eps).float()
        if zero_mask.sum() > 0:
            term1 = -(x_tilde**2) * lam / (1 + 2 * W * lam + self.eps)
            term2 = -0.5 * torch.log(1 + 2 * W * lam + self.eps)
            log_prob_zero = term1 + term2
            log_prob += zero_mask * log_prob_zero
        
        nonzero_mask = (y >= self.eps).float()
        if nonzero_mask.sum() > 0:
            term1 = -(y - x_tilde)**2 / (2 * W + self.eps)
            term2 = -0.5 * math.log(2 * math.pi) - 0.5 * torch.log(W + self.eps)
            term3 = torch.log(1 - torch.exp(-lam * (y**2)) + self.eps)
            log_prob_nonzero = term1 + term2 + term3
            log_prob += nonzero_mask * log_prob_nonzero
        
        return log_prob
    
    def model(self, y):
        pyro.module("decoder", self.decoder)
        batch_size = y.size(0)
        
        with pyro.plate("data", batch_size):
            z_loc = y.new_zeros([batch_size, self.n_latent])
            z_scale = y.new_ones([batch_size, self.n_latent])
            z = pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))
            
            x_tilde, W, lam = self.decoder(z)
            
            log_prob = self._compute_log_prob(y, x_tilde, W, lam)
            pyro.factor("obs", log_prob.sum(-1))
    
    def guide(self, y):
        self._init_variational_params(y.size(0))
        pyro.module("variational_params", self.variational_params)
        
        with pyro.plate("data", y.size(0)) as idx:
            mu_z, sigma_z = self.variational_params(idx)
            pyro.sample("latent", dist.Normal(mu_z, sigma_z).to_event(1))
    
    def get_latent(self, y=None):
        with torch.no_grad():
            if self.variational_params is None:
                raise ValueError("Model not trained yet")
            mu_z, _ = self.variational_params()
        return mu_z
    
    @property
    def device(self):
        return next(self.parameters()).device


def _fit_pyro_model(Y, K, threshold=float("-inf"), loss_threshold=float("-inf"), batch_size=128, lr=5e-4, non_linear=True, 
             Y_test=None, eval_train=True, method='amortized', max_epochs=100000,
             act_nng_exp=True, eps=1e-6):

    if method not in ['amortized', 'vi']:
        raise ValueError("method must be either 'amortized' or 'vi'")
    
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    N, D = Y.shape
    Y_tensor = torch.from_numpy(Y).float().to(device)
    
    if method == 'amortized':
        model = ZIFA_Amortized(D, K, non_linear=non_linear, act_nng_exp=act_nng_exp, eps=eps).to(device)
    else:
        model = ZIFA_VI(D, K, act_nng_exp=act_nng_exp, eps=eps).to(device)
    
    use_test = (method == 'amortized' and Y_test is not None)
    if method == 'vi' and Y_test is not None:
        print("Warning: method='vi' does not support test set evaluation. Ignoring Y_test.")
    
    optimizer = Adam({"lr": lr})
    svi = SVI(model.model, model.guide, optimizer, loss=Trace_ELBO())

    losses, epoches, times, losses_test = [], [], [], []
    step_losses, step_times, step_indices = [], [], []
    global_step = 0

    print(f"Training {method} method... Max epochs: {max_epochs}")
    
    converged = False
    convergence_epoch = None
    pre_loss = float('inf')
    offset_train = torch.mean(torch.sum(Y_tensor, dim=-1)).item()
    updates_per_epoch = max(1, int(N / batch_size))
    
    for epoch in range(max_epochs):
        epoch_losses = []
        perm = torch.randperm(N)
        for i in range(updates_per_epoch):
            if method == 'amortized':
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, N)
                idx = perm[start_idx:end_idx]
                batch_y = Y_tensor[idx]

                loss = svi.step(batch_y)
                global_step += 1
                step_loss = loss / batch_size
                step_losses.append(step_loss)
                step_times.append(time.time() - t0)
                step_indices.append(global_step)
            else:
                loss = svi.step(Y_tensor)
                global_step += 1

                step_loss = loss / N
                step_losses.append(step_loss)
                step_times.append(time.time() - t0)
                step_indices.append(global_step)
        
            epoch_losses.append(step_loss)
                
        step_loss_mean = np.mean(epoch_losses)
 
        if use_test:
            Y_test_tensor = torch.from_numpy(Y_test).float().to(device)
            offset_test = np.mean(np.sum(Y_test, axis=-1))
            test_loss = svi.evaluate_loss(Y_test_tensor) / Y_test.shape[0] + offset_test
            losses_test.append(test_loss)
        
        if np.abs(pre_loss - step_loss_mean) < threshold:
            converged = True
            convergence_epoch = epoch + 1
            epoches.append(epoch + 1)
            losses.append(step_loss_mean)
            times.append(time.time() - t0)
            Y_loss = svi.step(Y_tensor)
            X_Loss = Y_loss + offset_train
            print(f'Converged at epoch {epoch + 1}, Average Loss: {step_loss_mean:.6f}, X Loss: { X_Loss:.6f},Time: {times[-1]}')            
            break

        if step_loss_mean < loss_threshold:
            converged = True
            convergence_epoch = epoch + 1
            epoches.append(epoch + 1)
            losses.append(step_loss_mean)
            times.append(time.time() - t0)
            X_Loss = step_loss_mean + offset_train
            print(f'Converged at epoch {epoch + 1}, Average Loss: {step_loss_mean:.6f}, X Loss: { X_Loss:.6f},Time: {times[-1]}')            
            break
    
        pre_loss = step_loss_mean
        epoches.append(epoch + 1)
        losses.append(step_loss_mean)
        times.append(time.time() - t0)
    
        if (epoch + 1) % 100 == 0:
            print(f'Epoch {epoch + 1}/{max_epochs}, Average Loss: {step_loss_mean:.6f}, '
                  f'Total steps: {global_step}')

    if not converged:
        print(f'Reached max epochs ({max_epochs}). Final loss: {step_loss_mean:.6f}')

    result, result_test = {}, {}
    with torch.no_grad():
        if method == 'amortized':
            latent = model.get_latent(Y_tensor).cpu().numpy()
        else:
            latent = model.get_latent().cpu().numpy()
        
        A_c = model.decoder.W_netp.cpu().numpy()
        mu_c = model.decoder.b_netp.cpu().numpy().reshape([D, 1])
        W_c = torch.exp(model.decoder.logW).cpu().numpy().reshape([D, 1])
        lam_c = torch.exp(model.decoder.loglam).cpu().numpy()[0]
        
        result.update({
            "loss": step_loss_mean, 
            "latent": latent,
            "W": W_c, 
            "A": A_c, 
            "mu": mu_c,
            "lam": lam_c,
            "losses": losses, 
            "epochs": epoches,
            "times": times,
            "step_losses": step_losses,
            "step_times": step_times,
            "step_indices": step_indices,
            "total_steps": global_step,
            "method": method,
            "converged": converged,
            "convergence_epoch": convergence_epoch,
        })

        if use_test:
            Y_test_tensor = torch.from_numpy(Y_test).float().to(device)
            latent_test = model.get_latent(Y_test_tensor).cpu().numpy()
            result_test.update({
                "loss": losses_test[-1] if losses_test else None,
                "latent": latent_test,
                "losses": losses_test
            })
            
    return result, result_test


class ZIFA:

    _METHODS = {"classic", "block", "pyro"}

    def __init__(
        self,
        K: int,
        method: str = "classic",
        inference: str = "amortized",
        singleSigma: bool = False,
        n_blocks: Optional[int] = None,
        p0_thresh: float = 0.95,
        threshold: float = float("-inf"),
        loss_threshold: float = float("-inf"),
        batch_size: int = 128,
        lr: float = 5e-4,
        non_linear: bool = True,
        eval_train: bool = True,
        max_epochs: int = 100000,
        act_nng_exp: bool = True,
        eps: float = 1e-6,
    ) -> None:
        if method not in self._METHODS:
            raise ValueError(f"method must be one of {sorted(self._METHODS)}")
        if inference not in {"amortized", "vi"}:
            raise ValueError("inference must be 'amortized' or 'vi'")
        self.K = K
        self.method = method
        self.inference = inference
        self.singleSigma = singleSigma
        self.n_blocks = n_blocks
        self.p0_thresh = p0_thresh
        self.threshold = threshold
        self.loss_threshold = loss_threshold
        self.batch_size = batch_size
        self.lr = lr
        self.non_linear = non_linear
        self.eval_train = eval_train
        self.max_epochs = max_epochs
        self.act_nng_exp = act_nng_exp
        self.eps = eps
        self.result: Optional[Dict[str, Any]] = None
        self.result_test: Optional[Dict[str, Any]] = None

    def fit(self, Y, Y_test=None) -> "ZIFA":
        if self.method == "classic":
            self.result, self.result_test = _fit_classic_model(
                Y,
                self.K,
                singleSigma=self.singleSigma,
                Y_test=Y_test,
            )
        elif self.method == "block":
            self.result = _fit_block_model(
                Y,
                self.K,
                singleSigma=self.singleSigma,
                n_blocks=self.n_blocks,
                p0_thresh=self.p0_thresh,
            )
            self.result_test = {}
        else:
            self.result, self.result_test = _fit_pyro_model(
                Y=Y,
                K=self.K,
                threshold=self.threshold,
                loss_threshold=self.loss_threshold,
                batch_size=self.batch_size,
                lr=self.lr,
                non_linear=self.non_linear,
                Y_test=Y_test,
                eval_train=self.eval_train,
                method=self.inference,
                max_epochs=self.max_epochs,
                act_nng_exp=self.act_nng_exp,
                eps=self.eps,
            )
        return self

    @property
    def latent(self):
        if self.result is None:
            raise ValueError("Model has not been fit yet.")
        return self.result["latent"]
