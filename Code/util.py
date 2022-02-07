# -*- coding: utf-8 -*-
# @Date    : 2019-02-19 16:17:09
# @Author  : Your Name (you@example.org)
# @Link    : http://example.org
# @Version : $1.0$

import os, sys, csv
import random

import torch
import numpy as np
import pandas as pd

import cv2

from sklearn.metrics import pairwise_kernels
from sklearn.metrics import pairwise_distances
from torch.utils.data import Subset
from pandas.api.types import CategoricalDtype
from torch.autograd import Variable

SQRT_CONST = 1e-10


def mmd2_rbf(X,Y,sig=1.0):
    """ Computes the l2-RBF MMD for X Y """

    Kxx = np.exp(-pdist2sq(X,X)/np.square(sig))
    Kxy = np.exp(-pdist2sq(X,Y)/np.square(sig))
    Kyy = np.exp(-pdist2sq(Y,Y)/np.square(sig))

    m = np.float(X.shape[0])
    n = np.float(Y.shape[0])

    mmd = 1.0 /(m*(m-1.0))*(Kxx.sum()-m)
    mmd = mmd + 1.0 /(n*(n-1.0))*(Kyy.sum()-n)
    mmd = mmd - 2.0 /(m*n)*Kxy.sum()
#     mmd = 4.0*mmd

    return mmd

def pdist2sq(X,Y):
    """ Computes the squared Euclidean distance between all pairs x in X, y in Y """
    C = -2*np.matmul(X,Y.T)
    nx = np.sum(np.square(X),axis=1,keepdims=True)
    ny = np.sum(np.square(Y),axis=1,keepdims=True)
    D = (C + ny.T) + nx
    return D

def pdist2(X,Y):
    """ Returns the tensorflow pairwise distance matrix """
    return safe_sqrt(pdist2sq(X,Y))

def safe_sqrt(x, lbound=SQRT_CONST):
    ''' Numerically safe version of TensorFlow sqrt '''
    return np.sqrt(np.clip(x, lbound, np.inf))

def MMD2u(K, m, n):
    """The MMD^2_u unbiased statistic.
    """
    Kx = K[:m, :m]
    Ky = K[m:, m:]
    Kxy = K[:m, m:]
    return 1.0 / (m * (m - 1.0)) * (Kx.sum() - Kx.diagonal().sum()) + \
        1.0 / (n * (n - 1.0)) * (Ky.sum() - Ky.diagonal().sum()) - \
        2.0 / (m * n) * Kxy.sum()


def compute_null_distribution(K, m, n, iterations=10000, verbose=False,
                              random_state=None, marker_interval=1000):
    """Compute the bootstrap null-distribution of MMD2u.
    """
    if type(random_state) == type(np.random.RandomState()):
        rng = random_state
    else:
        rng = np.random.RandomState(random_state)

    mmd2u_null = np.zeros(iterations)
    for i in range(iterations):
        if verbose and (i % marker_interval) == 0:
            print(i),
            sys.stdout.flush()
        idx = rng.permutation(m+n)
        K_i = K[idx, idx[:, None]]
        mmd2u_null[i] = MMD2u(K_i, m, n)

    if verbose:
        print("")

    return mmd2u_null

def K_gen(X,Y,kernel_function='rbf'):
	X = X / safe_sqrt(np.sum(np.square(X), axis=1,keepdims=True))
	Y = Y / safe_sqrt(np.sum(np.square(Y), axis=1,keepdims=True))
	m = len(X)
	n = len(Y)
	XY = np.vstack([X, Y])
	K = pairwise_kernels(XY, metric=kernel_function)
	return K,m,n

def MMD_measure(X,Y,h0test=False):
    K,m,n = K_gen(X,Y)
    T_mmd2 = MMD2u(K, m, n)
    if h0test == True:
        iterations=10000
        mmd2u_null = compute_null_distribution(K,m,n,iterations)
        p = max(1.0/iterations, (mmd2u_null > T_mmd2).sum() /
                  float(iterations))
        print('p mmd2u: %f, t mmd2u: %f' % (p,T_mmd2))
    return T_mmd2

def set_seed(seed):
    """Sets seed"""
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def minimum(numbers, empty_val=0.):
    if isinstance(numbers, torch.Tensor):
        if numbers.numel()==0:
            return torch.tensor(empty_val, device=numbers.device)
        else:
            return numbers[~torch.isnan(numbers)].min()
    elif isinstance(numbers, np.ndarray):
        if numbers.size==0:
            return np.array(empty_val)
        else:
            return np.nanmin(numbers)
    else:
        if len(numbers)==0:
            return empty_val
        else:
            return min(numbers)

def maximum(numbers, empty_val=0.):
    if isinstance(numbers, torch.Tensor):
        if numbers.numel()==0:
            return torch.tensor(empty_val, device=numbers.device)
        else:
            return numbers[~torch.isnan(numbers)].max()
    elif isinstance(numbers, np.ndarray):
        if numbers.size==0:
            return np.array(empty_val)
        else:
            return np.nanmax(numbers)
    else:
        if len(numbers)==0:
            return empty_val
        else:
            return max(numbers)

def split_into_groups(g):
    """
    Args:
        - g (Tensor): Vector of groups
    Returns:
        - groups (Tensor): Unique groups present in g
        - group_indices (list): List of Tensors, where the i-th tensor is the indices of the
                                elements of g that equal groups[i].
                                Has the same length as len(groups).
        - unique_counts (Tensor): Counts of each element in groups.
                                 Has the same length as len(groups).
    """
    unique_groups, unique_counts = torch.unique(g, sorted=False, return_counts=True)
    group_indices = []
    for group in unique_groups:
        group_indices.append(
            torch.nonzero(g == group, as_tuple=True)[0])
    return unique_groups, group_indices, unique_counts

def get_counts(g, n_groups):
    """
    This differs from split_into_groups in how it handles missing groups.
    get_counts always returns a count Tensor of length n_groups,
    whereas split_into_groups returns a unique_counts Tensor
    whose length is the number of unique groups present in g.
    Args:
        - g (Tensor): Vector of groups
    Returns:
        - counts (Tensor): A list of length n_groups, denoting the count of each group.
    """
    unique_groups, unique_counts = torch.unique(g, sorted=False, return_counts=True)
    counts = torch.zeros(n_groups, device=g.device)
    counts[unique_groups] = unique_counts.float()
    return counts

def avg_over_groups(v, g, n_groups):
    """
    Args:
        v (Tensor): Vector containing the quantity to average over.
        g (Tensor): Vector of the same length as v, containing group information.
    Returns:
        group_avgs (Tensor): Vector of length num_groups
        group_counts (Tensor)
    """
    import torch_scatter
    assert v.device==g.device
    assert v.numel()==g.numel()
    group_count = get_counts(g, n_groups)
    group_avgs = torch_scatter.scatter(src=v, index=g, dim_size=n_groups, reduce='mean')
    return group_avgs, group_count

def map_to_id_array(df, ordered_map={}):
    maps = {}
    array = np.zeros(df.shape)
    for i, c in enumerate(df.columns):
        if c in ordered_map:
            category_type = CategoricalDtype(categories=ordered_map[c], ordered=True)
        else:
            category_type = 'category'
        series = df[c].astype(category_type)
        maps[c] = series.cat.categories.values
        array[:,i] = series.cat.codes.values
    return maps, array

def subsample_idxs(idxs, num=5000, take_rest=False, seed=None):
    seed = (seed + 541433) if seed is not None else None
    rng = np.random.default_rng(seed)

    idxs = idxs.copy()
    rng.shuffle(idxs)
    if take_rest:
        idxs = idxs[num:]
    else:
        idxs = idxs[:num]
    return idxs

def shuffle_arr(arr, seed=None):
    seed = (seed + 548207) if seed is not None else None
    rng = np.random.default_rng(seed)

    arr = arr.copy()
    rng.shuffle(arr)
    return arr

def threshold_at_recall(y_pred, y_true, global_recall=60):
    """ Calculate the model threshold to use to achieve a desired global_recall level. Assumes that
    y_true is a vector of the true binary labels."""
    return np.percentile(y_pred[y_true == 1], 100-global_recall)

def numel(obj):
    if torch.is_tensor(obj):
        return obj.numel()
    elif isinstance(obj, list):
        return len(obj)
    else:
        raise TypeError("Invalid type for numel")

def save_model(algorithm, epoch, best_val_metric, path):
    state = {}
    state['epoch'] = epoch
    state['best_val_metric'] = best_val_metric
    torch.save(state, path)

def tv_norm(input, tv_beta):
    img = input[0, 0, :]
    row_grad = torch.mean(torch.abs((img[:-1 , :] - img[1 :, :])).pow(tv_beta))
    col_grad = torch.mean(torch.abs((img[: , :-1] - img[: , 1 :])).pow(tv_beta))
    return row_grad + col_grad

def preprocess_image(img):
    means=[0.485, 0.456, 0.406]
    stds=[0.229, 0.224, 0.225]

    preprocessed_img = img.copy()[: , :, ::-1]
    for i in range(3):
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]
    preprocessed_img = \
        np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))

    # if use_cuda:
    preprocessed_img_tensor = torch.from_numpy(preprocessed_img).cuda()
    # else:
    #     preprocessed_img_tensor = torch.from_numpy(preprocessed_img)

    preprocessed_img_tensor.unsqueeze_(0)
    return Variable(preprocessed_img_tensor, requires_grad = False)

def save(mask, img, blurred):
    mask = mask.cpu().data.numpy()[0]
    mask = np.transpose(mask, (1, 2, 0))

    mask = (mask - np.min(mask)) / np.max(mask)
    mask = 1 - mask
    heatmap = cv2.applyColorMap(np.uint8(255*mask), cv2.COLORMAP_JET)

    heatmap = np.float32(heatmap) / 255
    cam = 1.0*heatmap + np.float32(img)/255
    cam = cam / np.max(cam)

    img = np.float32(img) / 255
    perturbated = np.multiply(1 - mask, img) + np.multiply(mask, blurred)
    
    return np.uint8(255*perturbated), np.uint8(255*heatmap), np.uint8(255*mask), np.uint8(255*cam)

def numpy_to_torch(img, requires_grad = True):
    if len(img.shape) < 3:
        output = np.float32([img])
    else:
        output = np.transpose(img, (2, 0, 1))

    output = torch.from_numpy(output)
    # if use_cuda:
    output = output.cuda()

    output.unsqueeze_(0)
    v = Variable(output, requires_grad = requires_grad)
    return v