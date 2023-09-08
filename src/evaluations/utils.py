import torch
from src.utils import *
from src.evaluations.augmentations import apply_augmentations, parse_augmentations, Basepoint, Scale
import signatory
import numpy as np
import math

def cov_torch(x, rowvar=False, bias=True, ddof=None, aweights=None):
    # """Estimates covariance matrix like numpy.cov"""
    # reshape x
    # _, T, C = x.shape
    # x = x.reshape(-1, T * C)
    # ensure at least 2D
    # if x.dim() == 1:
    #    x = x.view(-1, 1)

    # treat each column as a data point, each row as a variable
    # if rowvar and x.shape[0] != 1:
    #    x = x.t()

    # if ddof is None:
    #    if bias == 0:
    #        ddof = 1
    #    else:
    #        ddof = 0

    # w = aweights
    # if w is not None:
    #    if not torch.is_tensor(w):
    #        w = torch.tensor(w, dtype=torch.float)
    #    w_sum = torch.sum(w)
    #    avg = torch.sum(x * (w / w_sum)[:, None], 0)
    # else:
    #    avg = torch.mean(x, 0)

    # Determine the normalization
    # if w is None:
    #    fact = x.shape[0] - ddof
    # elif ddof == 0:
    #    fact = w_sum
    # elif aweights is None:
    #    fact = w_sum - ddof
    # else:
    #    fact = w_sum - ddof * torch.sum(w * w) / w_sum

    # xm = x.sub(avg.expand_as(x))

    # if w is None:
    #    X_T = xm.t()
    # else:
    #    X_T = torch.mm(torch.diag(w), xm).t()

    # c = torch.mm(X_T, xm)
    # c = c / fact

    # return c.squeeze()
    device = x.device
    x = to_numpy(x)
    _, L, C = x.shape
    x = x.reshape(-1, L*C)
    return torch.from_numpy(np.cov(x, rowvar=False)).to(device).float()

def q_var_torch(x: torch.Tensor):
    """
    :param x: torch.Tensor [B, S, D]
    :return: quadratic variation of x. [B, D]
    """
    return torch.sum(torch.pow(x[:, 1:] - x[:, :-1], 2), 1)


def acf_torch(x: torch.Tensor, max_lag: int, dim: Tuple[int] = (0, 1)) -> torch.Tensor:
    """
    :param x: torch.Tensor [B, S, D]
    :param max_lag: int. specifies number of lags to compute the acf for
    :return: acf of x. [max_lag, D]
    """
    acf_list = list()
    x = x - x.mean((0, 1))
    std = torch.var(x, unbiased=False, dim=(0, 1))
    for i in range(max_lag):
        y = x[:, i:] * x[:, :-i] if i > 0 else torch.pow(x, 2)
        acf_i = torch.mean(y, dim) / std
        acf_list.append(acf_i)
    if dim == (0, 1):
        return torch.stack(acf_list)
    else:
        return torch.cat(acf_list, 1)


def non_stationary_acf_torch(X, symmetric=False):
    """
    Compute the correlation matrix between any two time points of the time series
    Parameters
    ----------
    X (torch.Tensor): [B, T, D]
    symmetric (bool): whether to return the upper triangular matrix of the full matrix

    Returns
    -------
    Correlation matrix of the shape [T, T, D] where each entry (t_i, t_j, d_i) is the correlation between the d_i-th coordinate of X_{t_i} and X_{t_j}
    """
    # Get the batch size, sequence length, and input dimension from the input tensor
    B, T, D = X.shape

    # Create a tensor to hold the correlations
    correlations = torch.zeros(T, T, D)

    # Loop through each time step from lag to T-1
    for t in range(T):
        # Loop through each lag from 1 to lag
        for tau in range(t, T):
            # Compute the correlation between X_{t, d} and X_{t-tau, d}
            correlation = torch.sum(X[:, t, :] * X[:, tau, :], dim=0) / (
                torch.norm(X[:, t, :], dim=0) * torch.norm(X[:, tau, :], dim=0))
            # print(correlation)
            # Store the correlation in the output tensor
            correlations[t, tau, :] = correlation
            if symmetric:
                correlations[tau, t, :] = correlation

    return correlations


def cacf_torch(x, lags: list, dim=(0, 1)):
    """
    Computes the cross-correlation between feature dimension and time dimension
    Parameters
    ----------
    x
    lags
    dim

    Returns
    -------

    """
    # Define a helper function to get the lower triangular indices for a given dimension
    def get_lower_triangular_indices(n):
        return [list(x) for x in torch.tril_indices(n, n)]

    # Get the lower triangular indices for the input tensor x
    ind = get_lower_triangular_indices(x.shape[2])

    # Standardize the input tensor x along the given dimensions
    x = (x - x.mean(dim, keepdims=True)) / x.std(dim, keepdims=True)

    # Split the input tensor into left and right parts based on the lower triangular indices
    x_l = x[..., ind[0]]
    x_r = x[..., ind[1]]

    # Compute the cross-correlation at each lag and store in a list
    cacf_list = list()
    for i in range(lags):
        # Compute the element-wise product of the left and right parts, shifted by the lag if i > 0
        y = x_l[:, i:] * x_r[:, :-i] if i > 0 else x_l * x_r

        # Compute the mean of the product along the time dimension
        cacf_i = torch.mean(y, (1))

        # Append the result to the list of cross-correlations
        cacf_list.append(cacf_i)

    # Concatenate the cross-correlations across lags and reshape to the desired output shape
    cacf = torch.cat(cacf_list, 1)
    return cacf.reshape(cacf.shape[0], -1, len(ind[0]))


def rmse(x, y):
    return (x - y).pow(2).sum().sqrt()


def skew_torch(x, dim=(0, 1), dropdims=True):
    x = x - x.mean(dim, keepdims=True)
    x_3 = torch.pow(x, 3).mean(dim, keepdims=True)
    x_std_3 = torch.pow(x.std(dim, unbiased=True, keepdims=True), 3)
    skew = x_3 / x_std_3
    if dropdims:
        skew = skew[0, 0]
    return skew


def kurtosis_torch(x, dim=(0, 1), excess=True, dropdims=True):
    x = x - x.mean(dim, keepdims=True)
    x_4 = torch.pow(x, 4).mean(dim, keepdims=True)
    x_var2 = torch.pow(torch.var(x, dim=dim, unbiased=False, keepdims=True), 2)
    kurtosis = x_4 / x_var2
    if excess:
        kurtosis = kurtosis - 3
    if dropdims:
        kurtosis = kurtosis[0, 0]
    return kurtosis


def diff(x): return x[:, 1:] - x[:, :-1]
def acf_diff(x): return torch.sqrt(torch.pow(x, 2).sum(0))
def cc_diff(x): return torch.abs(x).sum(0)
def cov_diff(x): return torch.abs(x).mean()

def histogram_torch(x, n_bins, density=True):
    a, b = x.min().item(), x.max().item()
    b = b+1e-5 if b == a else b
    # delta = (b - a) / n_bins
    bins = torch.linspace(a, b, n_bins+1)
    delta = bins[1]-bins[0]
    # bins = torch.arange(a, b + 1.5e-5, step=delta)
    count = torch.histc(x, bins=n_bins, min=a, max=b).float()
    if density:
        count = count / delta / float(x.shape[0] * x.shape[1])
    return count, bins



def get_gbm(size, n_lags, d=1, drift=0., scale=0.1, h=1):
    x_real = torch.ones(size, n_lags, d)
    x_real[:, 1:, :] = torch.exp(
        (drift-scale**2/2)*h + (scale*np.sqrt(h)*torch.randn(size, n_lags-1, d)))
    x_real = x_real.cumprod(1)
    return x_real


class SigUtils:

    @staticmethod
    def compute_expected_signature(x_path, depth: int, augmentations: Tuple, normalise: bool = True):
        x_path_augmented = apply_augmentations(x_path, augmentations)
        expected_signature = signatory.signature(
            x_path_augmented, depth=depth).mean(0)
        dim = x_path_augmented.shape[2]
        count = 0
        if normalise:
            for i in range(depth):
                expected_signature[count:count + dim**(
                    i+1)] = expected_signature[count:count + dim**(i+1)] * math.factorial(i+1)
                count = count + dim**(i+1)
        return expected_signature