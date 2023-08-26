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
from src.evaluations.utils import *
from src.evaluations.metrics import *

class Loss(nn.Module):
    def __init__(self, name, reg=1.0, transform=lambda x: x, threshold=10., backward=False, norm_foo=lambda x: x):
        super(Loss, self).__init__()
        self.name = name
        self.reg = reg
        self.transform = transform
        self.threshold = threshold
        self.backward = backward
        self.norm_foo = norm_foo

    def forward(self, x_fake):
        self.loss_componentwise = self.compute(x_fake)
        return self.reg * self.loss_componentwise.mean()

    def compute(self, x_fake):
        raise NotImplementedError()

    @property
    def success(self):
        return torch.all(self.loss_componentwise <= self.threshold)


class ACFLoss(Loss):
    def __init__(self, x_real, max_lag=64, stationary=True, **kwargs):
        super(ACFLoss, self).__init__(norm_foo=acf_diff, **kwargs)
        self.max_lag = min(max_lag, x_real.shape[1])
        self.stationary = stationary
        if stationary:
            self.acf_real = acf_torch(self.transform(
                x_real), self.max_lag, dim=(0, 1))
        else:
            self.acf_real = non_stationary_acf_torch(self.transform(
                x_real), symmetric=False)  # Divide by 2 because it is symmetric matrix

    def compute(self, x_fake):
        if self.stationary:
            acf_fake = acf_torch(self.transform(x_fake), self.max_lag)
        else:
            acf_fake = non_stationary_acf_torch(self.transform(
                x_fake), symmetric=False)
        return self.norm_foo(acf_fake - self.acf_real.to(x_fake.device))


class MeanLoss(Loss):
    def __init__(self, x_real, **kwargs):
        super(MeanLoss, self).__init__(norm_foo=torch.abs, **kwargs)
        self.mean = x_real.mean((0, 1))

    def compute(self, x_fake, **kwargs):
        return self.norm_foo(x_fake.mean((0, 1)) - self.mean)


class StdLoss(Loss):
    def __init__(self, x_real, **kwargs):
        super(StdLoss, self).__init__(norm_foo=torch.abs, **kwargs)
        self.std_real = x_real.std((0, 1))

    def compute(self, x_fake, **kwargs):
        return self.norm_foo(x_fake.std((0, 1)) - self.std_real)


class SkewnessLoss(Loss):
    def __init__(self, x_real, **kwargs):
        super(SkewnessLoss, self).__init__(norm_foo=torch.abs, **kwargs)
        self.skew_real = skew_torch(self.transform(x_real))

    def compute(self, x_fake, **kwargs):
        skew_fake = skew_torch(self.transform(x_fake))
        return self.norm_foo(skew_fake - self.skew_real)


class KurtosisLoss(Loss):
    def __init__(self, x_real, **kwargs):
        super(KurtosisLoss, self).__init__(norm_foo=torch.abs, **kwargs)
        self.kurtosis_real = kurtosis_torch(self.transform(x_real))

    def compute(self, x_fake):
        kurtosis_fake = kurtosis_torch(self.transform(x_fake))
        return self.norm_foo(kurtosis_fake - self.kurtosis_real)


class CrossCorrelLoss(Loss):
    def __init__(self, x_real, max_lag=64, **kwargs):
        super(CrossCorrelLoss, self).__init__(norm_foo=cc_diff, **kwargs)
        self.cross_correl_real = cacf_torch(
            self.transform(x_real), max_lag).mean(0)[0]
        self.max_lag = max_lag

    def compute(self, x_fake):
        cross_correl_fake = cacf_torch(
            self.transform(x_fake), self.max_lag).mean(0)[0]
        loss = self.norm_foo(
            cross_correl_fake - self.cross_correl_real.to(x_fake.device)).unsqueeze(0)
        return loss


class HistoLoss(Loss):

    def __init__(self, x_real, n_bins, **kwargs):
        super(HistoLoss, self).__init__(**kwargs)
        self.densities = list()
        self.locs = list()
        self.deltas = list()
        for i in range(x_real.shape[2]):
            tmp_densities = list()
            tmp_locs = list()
            tmp_deltas = list()
            # Exclude the initial point
            for t in range(x_real.shape[1]):
                x_ti = x_real[:, t, i].reshape(-1, 1)
                d, b = histogram_torch(x_ti, n_bins, density=True)
                tmp_densities.append(nn.Parameter(d).to(x_real.device))
                delta = b[1:2] - b[:1]
                loc = 0.5 * (b[1:] + b[:-1])
                tmp_locs.append(loc)
                tmp_deltas.append(delta)
            self.densities.append(tmp_densities)
            self.locs.append(tmp_locs)
            self.deltas.append(tmp_deltas)

    def compute(self, x_fake):
        loss = list()

        def relu(x):
            return x * (x >= 0.).float()

        for i in range(x_fake.shape[2]):
            tmp_loss = list()
            # Exclude the initial point
            for t in range(x_fake.shape[1]):
                loc = self.locs[i][t].view(1, -1).to(x_fake.device)
                x_ti = x_fake[:, t, i].contiguous(
                ).view(-1, 1).repeat(1, loc.shape[1])
                dist = torch.abs(x_ti - loc)
                counter = (relu(self.deltas[i][t].to(
                    x_fake.device) / 2. - dist) > 0.).float()
                density = counter.mean(0) / self.deltas[i][t].to(x_fake.device)
                abs_metric = torch.abs(
                    density - self.densities[i][t].to(x_fake.device))
                loss.append(torch.mean(abs_metric, 0))
        loss_componentwise = torch.stack(loss)
        return loss_componentwise


class CovLoss(Loss):
    def __init__(self, x_real, **kwargs):
        super(CovLoss, self).__init__(norm_foo=cov_diff, **kwargs)
        self.covariance_real = cov_torch(
            self.transform(x_real))

    def compute(self, x_fake):
        covariance_fake = cov_torch(self.transform(x_fake))
        loss = self.norm_foo(covariance_fake -
                             self.covariance_real.to(x_fake.device))
        return loss
    

class CrossCorrelation(Loss):
    def __init__(self, x_real, **kwargs):
        super(CrossCorrelation).__init__(**kwargs)
        self.x_real = x_real

    def compute(self, x_fake):
        fake_corre = torch.from_numpy(np.corrcoef(
            x_fake.mean(1).permute(1, 0))).float()
        real_corre = torch.from_numpy(np.corrcoef(
            self.x_real.mean(1).permute(1, 0))).float()
        return torch.abs(fake_corre-real_corre)
    


def FID_score(model, input_real, input_fake):
    """compute the FID score

    Args:
        model (torch model): pretrained rnn model
        input_real (torch.tensor):
        input_fake (torch.tensor):
    """

    device = input_real.device
    linear = model.to(device).linear1
    rnn = model.to(device).rnn
    act_real = linear(rnn(input_real)[
        0][:, -1]).detach().cpu().numpy()
    act_fake = linear(rnn(input_fake)[
        0][:, -1]).detach().cpu().numpy()
    mu_real = np.mean(act_real, axis=0)
    sigma_real = np.cov(act_real, rowvar=False)
    mu_fake = np.mean(act_fake, axis=0)
    sigma_fake = np.cov(act_fake, rowvar=False)
    return calculate_frechet_distance(mu_real, sigma_real, mu_fake, sigma_fake)



class Sig_MMD_loss(Loss):
    def __init__(self, x_real, depth, **kwargs):
        super(Sig_MMD_loss, self).__init__(**kwargs)
        self.x_real = x_real
        self.depth = depth

    def compute(self, x_fake):
        return Sig_mmd(self.x_real, x_fake, self.depth)


class Predictive_FID(Loss):
    def __init__(self, x_real, model, **kwargs):
        super(Predictive_FID, self).__init__(**kwargs)
        self.model = model
        self.x_real = x_real

    def compute(self, x_fake):
        return FID_score(self.model, self.x_real, x_fake)


class Predictive_KID(Loss):
    def __init__(self, x_real, model, **kwargs):
        super(Predictive_KID, self).__init__(**kwargs)
        self.model = model
        self.x_real = x_real

    def compute(self, x_fake):
        return KID_score(self.model, self.x_real, x_fake)
    


class SigW1Loss(Loss):
    def __init__(self, x_real, depth, **kwargs):
        name = kwargs.pop('name')
        super(SigW1Loss, self).__init__(name=name)
        self.sig_w1_metric = SigW1Metric(x_real=x_real, depth=depth, **kwargs)

    def compute(self, x_fake):
        loss = self.sig_w1_metric(x_fake)
        return loss

# W1 metric
class W1(Loss):
    def __init__(self, D, x_real, **kwargs):
        name = kwargs.pop('name')
        super(W1, self).__init__(name=name)
        self.D = D
        self.D_real = D(x_real).mean()

    def compute(self, x_fake):
        loss = self.D_real-self.D(x_fake).mean()
        return loss
