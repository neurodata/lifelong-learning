# Copyright 2019 NeuroData (http://neurodata.io)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Herein is some common code for Lifelong Forests. Some of this file
# contains code adapted from conditional entropy forests, which can
# be found here: 
# https://github.com/rguo123/conditional_entropy_forests/blob/master/code/algorithm.py

from sklearn.ensemble.forest import _generate_unsampled_indices
from sklearn.ensemble.forest import _generate_sample_indices
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import NearestNeighbors

from tqdm import tqdm

from itertools import permutations
import sys

import numpy as np

np.warnings.filterwarnings('ignore')

def finite_sample_correction(class_probs, row_sums):
    
    where_0 = np.argwhere(class_probs == 0)
    for elem in where_0:
        class_probs[elem[0], elem[1]] = 1 / (2 * row_sums[elem[0], None])
    where_1 = np.argwhere(class_probs == 1)
    for elem in where_1:
        class_probs[elem[0], elem[1]] = 1 - 1 / (2 * row_sums[elem[0], None])
    
    return class_probs

def build_model(X, y, n_estimators=200, max_samples=.32,
                                            bootstrap=True,
                                            depth=30,
                                            min_samples_leaf=1):
    if X.ndim == 1:
        raise ValueError('1d data will cause headaches down the road')
        
    max_features = int(np.ceil(np.sqrt(X.shape[1])))
        
    model=BaggingClassifier(DecisionTreeClassifier(max_depth=depth, min_samples_leaf=min_samples_leaf,
                                                     max_features = max_features),
                              n_estimators=n_estimators,
                              max_samples=max_samples,
                              bootstrap=bootstrap)
    
    model.fit(X, y)
    return model

def get_leaves(estimator):
    # adapted from https://scikit-learn.org/stable/auto_examples/tree/plot_unveil_tree_structure.html
    n_nodes = estimator.tree_.node_count
    children_left = estimator.tree_.children_left
    children_right = estimator.tree_.children_right
    feature = estimator.tree_.feature
    threshold = estimator.tree_.threshold
    
    leaf_ids = []
    stack = [(0, -1)] 
    while len(stack) > 0:
        node_id, parent_depth = stack.pop()

        # If we have a test node
        if (children_left[node_id] != children_right[node_id]):
            stack.append((children_left[node_id], parent_depth + 1))
            stack.append((children_right[node_id], parent_depth + 1))
        else:
            leaf_ids.append(node_id)
            
    return np.array(leaf_ids)

def estimate_posteriors(model, train, y, test, in_task=True, subsample=1, acorn=None):
	if acorn is None:
	    acorn = np.random.randint(10**6)
	np.random.seed(acorn)

	n, d = train.shape
	m, d_ = test.shape

	if d != d_:
	    raise ValueError("train and test data in different dimensions")

	class_counts = np.zeros((m, model.n_classes_))
    unique_labels = np.unique(y)
    def _map(u, x):
        return np.where(u == x)[0][0]

	class_counts = np.zeros((m, model.n_classes_))
	for tree in model:
	    # get out of bag indicies
	    if in_task:
	        prob_indices = _generate_unsampled_indices(tree.random_state, n)
	        # in_bag_idx = _generate_sample_indices(tree.random_state, n) # this is not behaving as i expected
	    else:
	        prob_indices = np.random.choice(range(n), size=int(subsample*n), replace=False)
	    
	    leaf_nodes = get_leaves(tree)
	    unique_leaf_nodes = np.unique(leaf_nodes)
	        
	    # get all node counts
	    node_counts = tree.tree_.n_node_samples
	    # get probs for eval samples
	    posterior_class_counts = np.zeros((len(unique_leaf_nodes), model.n_classes_))

	    for prob_index in prob_indices:
	        temp_node = tree.apply(train[prob_index].reshape(1, -1)).item()
	        posterior_class_counts[np.where(unique_leaf_nodes == temp_node)[0][0], y[prob_index]] += 1
	        
	    # total number of points in a node
	    row_sums = posterior_class_counts.sum(axis=1)
	    
	    # no divide by zero
	    row_sums[row_sums == 0] = 1

	    # posteriors
	    class_probs = (posterior_class_counts / row_sums[:, None])
	    # posteriors with finite sampling correction
	    
	    class_probs = finite_sample_correction(class_probs, row_sums)

	    # posteriors as a list
	    class_probs.tolist()
	    
	    partition_counts = np.asarray([node_counts[np.where(unique_leaf_nodes == x)[0][0]] for x in tree.apply(test)])
	    # get probability for out of bag samples
	    eval_class_probs = [class_probs[np.where(unique_leaf_nodes == x)[0][0]] for x in tree.apply(test)]
	    eval_class_probs = np.array(eval_class_probs)
	    # find total elements for out of bag samples
	    elems = np.multiply(eval_class_probs, partition_counts[:, np.newaxis])
	    # store counts for each x (repeat fhis for each tree)
	    class_counts += elems
	# calculate p(y|X = x) for all x's
	probs = class_counts / class_counts.sum(axis=1, keepdims=True)

	return probs

def predict(a):
    return np.argmax(a, axis = 1)

def permutation(predict1, predict2, force=False):
    """
    how to use:
    
    this function returns the permutation i.e. \pi: [K] -> [K] that maximizes
    the number of matched predictions
    
    to use the permutation for posteriors for point i (posterior_i), say, simply use
    posterior_i[permutation]
    
    """
    unique_1 = np.unique(predict1)
    unique_1_new = np.arange(len(unique_1))
    
    unique_2 = np.unique(predict2)
    unique_2_new = np.arange(len(unique_2))
    
    if force:
        for i, u in enumerate(unique_1_new):
            if u not in unique_2_new:
                unique_2_new = np.concatenate((unique_2_new, [u]))
        
        for i, u in enumerate(unique_2_new):
            if u not in unique_1_new:
                unique_1_new = np.concatenate((unique_1_new, [u]))
    else:
        if set(unique_1) != set(unique_2):
            raise ValueError("predictions must be on the same set of labels")
        
    K = len(unique_1_new)
    
    max_sum = 0
    max_perm = unique_2_new
    for i, perm in enumerate(permutations(unique_2_new)):
        perm = np.array(list(perm))
        temp_predict2 = -1*np.ones(len(predict2))
        
        for k in range(K):
            temp_predict2[np.where(predict2 == unique_2_new[k])[0]] = perm[k]
           
        temp_sum = np.sum(predict1 == temp_predict2)
        if temp_sum > max_sum:
            max_sum = temp_sum
            max_perm = perm
            
    return max_perm

def permute(a, perm):
    unique_a = np.unique(a)
    new_a = np.zeros(len(a))
    for i, u in enumerate(unique_a):
        new_a[np.where(a == u)[0]] = perm[i]
    
    return new_a.astype(int)

def estimate_alpha(predict1, predict2, permutation=None):
    if permutation is None:
        return 2 * (( np.sum(predict1 == predict2) / len(predict1)) - 0.5)
    else:
        unique = np.unique(predict2)
        unique_temp = np.unique(predict1)
        if len(unique) != len(unique_temp):
            unique = np.concatenate((unique, range(len(unique), len(unique) + len(unique_temp) - len(unique_temp))))
        temp_predict2 = -1*np.ones(len(predict2))
        
        for i, k in enumerate(unique):
            print(np.where(predict2 == k)[0])
            print(permutation[i])
            temp_predict2[np.where(predict2 == k)[0]] = permutation[i]
            
        return 2 * (np.sum(predict1 == temp_predict2) / len(predict1) - 0.5)

# Now some functions to help generate data

def generate_parity(n, d=2, invert_labels=False,acorn=None):
    if acorn is not None:
        np.random.seed(acorn)
        
    X = np.random.uniform(-1, 1, size=(n, d))
    Y = (np.sum(X > 0, axis=1) % 2 == 0).astype(int)
    
    if invert_labels:
        Y = -1 * (Y - 1)
    
    return X, Y.astype(int)

def generate_box(n, d=2, invert_labels=False, acorn=None):
    if acorn is not None:
        np.random.seed(acorn)

    X = np.random.uniform(-1, 1, size=(n, d))
    
    Y = -1*np.ones(n)
    
    for i in range(n):
        if X[i, 0] > 3/4 or X[i, 0] < -3/4 or X[i, 1] > 3/4 or X[i, 1] < -3/4:
            Y[i]=1
        else:
            Y[i]=0
            
#     if invert_labels:
#         Y = -1 * (Y - 1)

    return X, Y.astype(int)

# Now a function to run transfer learning experiments

def transfer_learning_experiment(nx, nz, d, dist_x=generate_parity, dist_z=generate_parity, m=100, target="Z", subsample = 1, n_algos=6):
    if target == "Z":
        invert_z = True
        invert_x = False
    else:
        invert_z = False
        invert_x = True
        
    Tx = int(np.floor(np.sqrt(nx)))
    Tz = int(np.floor(np.sqrt(nz)))
    
    Kx = int(np.floor(np.log(nx)))
    Kz = int(np.floor(np.log(nz)))
    
    errors = np.zeros(n_algos)

    # Source task
    X, labelsX = dist_x(nx, d, invert_labels=invert_x)
    testX, test_labelsX = dist_x(m, d, invert_labels=invert_x)

    
    # Target task
    Z, labelsZ = dist_z(nz, d, invert_labels=invert_z)
    testZ, test_labelsZ = dist_z(m, d, invert_labels=invert_z)


    model_X = build_model(X, labelsX, Tx)
    model_Z = build_model(Z, labelsZ, Tz)

    posteriors_structX_estX=estimate_posteriors(model_X, X, labelsX, testX, in_task=True)
    posteriors_structZ_estX=estimate_posteriors(model_Z, X, labelsX, testX, in_task=False, subsample=subsample)

    pred_structX_estX=predict(posteriors_structX_estX)
    pred_structZ_estX=predict(posteriors_structZ_estX)

    posteriors_structX_estZ=estimate_posteriors(model_X, Z, labelsZ, testZ, in_task=False, subsample=subsample)
    posteriors_structZ_estZ=estimate_posteriors(model_Z, Z, labelsZ, testZ, in_task=True)
    
    pred_structX_estZ=predict(posteriors_structX_estZ)
    pred_structZ_estZ=predict(posteriors_structZ_estZ)

    # calculate errors without attempting to transfer knowledge
    pred_X = predict(posteriors_structX_estX)
    pred_Z = predict(posteriors_structZ_estZ)

    errors[0] = 1 - np.sum(test_labelsX == pred_X)/m
    errors[3] = 1 - np.sum(test_labelsZ == pred_Z)/m

    # jtv ?
    pred_X_jtv = predict(posteriors_structX_estX + posteriors_structZ_estX)
    pred_Z_jtv = predict(posteriors_structZ_estZ + posteriors_structX_estZ)

    errors[1] = 1 - np.sum(test_labelsX == pred_X_jtv)/m
    errors[4] = 1 - np.sum(test_labelsZ == pred_Z_jtv)/m
    
    # Sum
    X, labelsX = dist_x(nz + nx, d, invert_labels=invert_x)
    
    model_best_X = build_model(X, labelsX, int(np.floor(np.sqrt(nx + nz))))
    
    posteriors_best_X=estimate_posteriors(model_best_X, X, labelsX, testX, in_task=True)
    predictions_best_X=predict(posteriors_best_X)
    
    errors[2] = 1 - np.sum(test_labelsX == predictions_best_X)/m
    
    # Sum
    Z, labelsZ = dist_z(nz + nx, d, invert_labels=invert_z)
    
    model_best_Z = build_model(Z, labelsZ, int(np.floor(np.sqrt(nx + nz))))
    
    posteriors_best_Z=estimate_posteriors(model_best_Z, Z, labelsZ, testZ, in_task=True)
    predictions_best_Z=predict(posteriors_best_Z)
    
    errors[5] = 1 - np.sum(test_labelsZ == predictions_best_Z)/m

    if n_algos > 6:
	    # now using local estimates..
	    # need to debug more..
	    new_posteriors_structX_estZ_local = np.zeros(posteriors_structX_estZ.shape)
	    new_posteriors_structZ_estX_local = np.zeros(posteriors_structZ_estX.shape)
	    
	    train_pred_structX_estX = predict(estimate_posteriors(model_X, X, labelsX, X, in_task=True))
	    train_pred_structZ_estX = predict(estimate_posteriors(model_Z, Z, labelsZ, X, in_task=False))
	    
	    
	    train_pred_structZ_estZ = predict(estimate_posteriors(model_Z, Z, labelsZ, Z, in_task=True))
	    train_pred_structX_estZ = predict(estimate_posteriors(model_X, X, labelsX, Z, in_task=False))

	    kNN_X = NearestNeighbors(n_neighbors=Kx).fit(X)    
	    kNN_Z = NearestNeighbors(n_neighbors=Kz).fit(Z)

	    alpha_X_local = np.zeros(m)
	    for k, obs in enumerate(testX):
	        print(k)
	        obs = obs.reshape(1, -1)
	        temp_neighbors = kNN_X.kneighbors(obs)[1][0]
	        
	        temp_preds_structX_estX = train_pred_structX_estX[temp_neighbors]
	        temp_preds_structZ_estX = train_pred_structZ_estX[temp_neighbors]

	        temp_permutation = permutation(temp_preds_structX_estX, temp_preds_structZ_estX, force=True)
	        temp_permuted = permute(temp_preds_structZ_estX, temp_permutation)
	        
	        pred_structZ_estX_local = predict(new_posteriors_structZ_estX_local)    
	        alpha_X_local[k] = estimate_alpha(temp_preds_structX_estX, temp_permuted)
	        new_posteriors_structZ_estX_local[k] = alpha_X_local[k]*posteriors_structZ_estX[k][temp_permutation]

	    alpha_Z_local = np.zeros(m)
	    for k, obs in enumerate(testZ):
	        obs = obs.reshape(1, -1)
	        temp_neighbors = kNN_Z.kneighbors(obs)[1]

	        temp_preds_structX_estZ = train_pred_structX_estZ[temp_neighbors]
	        temp_preds_structZ_estZ = train_pred_structZ_estZ[temp_neighbors]

	        alpha_Z_local[k] = estimate_alpha(temp_preds_structZ_estZ, temp_preds_structX_estZ)
	        new_posteriors_structX_estZ_local[k] = alpha_Z_local[k]*posteriors_structX_estZ[k]

	    pred_X_cep_local = predict(posteriors_structX_estX + new_posteriors_structZ_estX_local)
	    pred_Z_cep_local = predict(posteriors_structZ_estZ + new_posteriors_structX_estZ_local)

	    errors[6] = 1 - np.sum(test_labelsX == pred_X_cep_local)/m
	    errors[7] = 1 - np.sum(test_labelsZ == pred_Z_cep_local)/m

    return errors