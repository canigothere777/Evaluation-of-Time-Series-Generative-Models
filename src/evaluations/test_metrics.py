from src.evaluations.augmentations import apply_augmentations, parse_augmentations, Basepoint, Scale
from functools import partial
from typing import Tuple, Optional
from src.utils import to_numpy
import math
# from .trainers.sig_wgan import SigW1Metric

import torch
from torch import nn
import numpy as np
from os import path as pt

import warnings
from scipy import linalg
from sklearn.metrics.pairwise import polynomial_kernel
# import signatory
import ksig
from src.utils import AddTime
import signatory

# from src.evaluations.loss import *
from src.evaluations.utils import *
from src.evaluations.loss import *

'''
List of test metrics:
- Sig MMD
- Sig W1
- Marginal distribution
- Cross correlation
- Covariance
'''

############### TEST METRICS #####################

test_metrics = {
    'Sig_mmd': partial(Sig_MMD_loss, name='Sig_mmd', depth=4),
    'SigW1': partial(SigW1Loss, name='SigW1', augmentations=[], normalise=False, depth=4),
    'marginal_distribution': partial(HistoLoss, n_bins=50, name='marginal_distribution'),
    'cross_correl': partial(CrossCorrelLoss, name='cross_correl'),
    'covariance': partial(CovLoss, name='covariance'),
    'auto_correl': partial(ACFLoss, name='auto_correl')
    }


def is_multivariate(x: torch.Tensor):
    """ Check if the path / tensor is multivariate. """
    return True if x.shape[-1] > 1 else False


def get_standard_test_metrics(x: torch.Tensor, **kwargs):
    """ Initialise list of standard test metrics for evaluating the goodness of the generator. """
    if 'model' in kwargs:
        model = kwargs['model']
    test_metrics_list = [test_metrics['Sig_mmd'](x),
                         test_metrics['SigW1'](x),
                         test_metrics['marginal_distribution'](x),
                         test_metrics['cross_correl'](x),
                         test_metrics['covariance'](x),
                         test_metrics['auto_correl'](x)
                         ]
    return test_metrics_list


def sig_mmd_permutation_test(X, Y, num_permutation) -> float:
    """two sample permutation test
    Args:
        test_func (function): function inputs: two batch of test samples, output: statistic
        X (torch.tensor): batch of samples (N,C) or (N,T,C)
        Y (torch.tensor): batch of samples (N,C) or (N,T,C)
        num_permutation (int):
    Returns:
        float: test power
    """
    # compute H1 statistics
    # test_func.eval()

    # We first split the data X into two subsets
    idx = torch.randint(X.shape[0], (X.shape[0],))

    X1 = X[idx[-int(X.shape[0]//2):]]
    X = X[idx[:-int(X.shape[0]//2)]]

    with torch.no_grad():

        t0 = Sig_mmd(X, X1, depth=5).cpu().detach().numpy()
        t1 = Sig_mmd(X, Y, depth=5).cpu().detach().numpy()
        print(t1)
        n, m = X.shape[0], Y.shape[0]
        combined = torch.cat([X, Y])

        statistics = []

        for i in range(num_permutation):
            idx1 = torch.randperm(n+m)

            statistics.append(
                Sig_mmd(combined[idx1[:n]], combined[idx1[n:]], depth=5))
            # print(statistics)
        # print(np.array(statistics))
    power = (t1 > torch.tensor(statistics).cpu(
    ).detach().numpy()).sum()/num_permutation
    type1_error = 1 - (t0 > torch.tensor(statistics).cpu(
    ).detach().numpy()).sum()/num_permutation
    return power, type1_error
