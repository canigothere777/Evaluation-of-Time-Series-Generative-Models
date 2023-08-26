from random import shuffle
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
import pandas as pd
from src.evaluations.test_metrics import Sig_mmd
from src.evaluations.evaluations import compute_discriminative_score, compute_predictive_score, sig_fid_model
from torch.utils.data import DataLoader, TensorDataset
from src.evaluations.test_metrics import Predictive_FID, Predictive_KID
from src.utils import to_numpy
from src.evaluations.utils import get_gbm
from src.datasets.AR1 import AROne


def permutation_test(test_func_arg_tuple,X, X1, Y, num_permutation) -> float:
    """two sample permutation test 

    Args:
        test_func (function): function inputs: two batch of test samples, 
        output: statistic
        X (torch.tensor): batch of samples (N,C) or (N,T,C)
        Y (torch.tensor): batch of samples (N,C) or (N,T,C)
        num_permutation (int): 
    Returns:
        float: test power
    """
    # compute H1 statistics
    # test_func.eval()
    test_func, kwargs = test_func_arg_tuple
    with torch.no_grad():

        t0 = to_numpy(test_func(X, X1,**kwargs))
        t1 = to_numpy(test_func(X, Y,**kwargs))

        n, m = X.shape[0], Y.shape[0]
        combined = torch.cat([X, Y])
        statistics = []

        for i in range(num_permutation):
            idx1 = torch.randperm(n+m)
            stat = test_func(combined[idx1[:n]], combined[idx1[n:]])
            statistics.append(stat)

    power = (t1 > to_numpy(torch.tensor(statistics))).sum()/num_permutation
    type1_error = 1 - (t0 > to_numpy(torch.tensor(statistics))).sum()/num_permutation
    return power, type1_error

def sig_mmd_permutation_test(X, X1, Y, num_permutation) -> float:
    test_func_arg_tuple = (Sig_mmd,{'depth':5})
    return permutation_test(test_func_arg_tuple,X, X1, Y, num_permutation)


class Compare_test_metrics:

    def __init__(self, X, Y, config):
        self.X = X
        self.Y = Y
        self.config = config

    @staticmethod
    def subsample(X, sample_size):
        if sample_size > X.shape[0]:
            raise ValueError('required samples size is larger than data size')
        idx = torch.randperm(X.shape[0])
        return X[idx[:sample_size]]

    @staticmethod
    def create_monotonic_dataset(X, X1, Y, i):
        #  replace Y by X up to dimension i, as i increases the dicrepency is smaller
        if i == 0:
            Y = X1  # when disturbance is 0, we use data from the same distribution as X, but not X
        else:
            Y[..., :-i] = X1[..., :-i]
        return X, Y
    
    def run_montontic_test_per_level(self, distubance_level: int, sample_size,num_cut = 2000):
       
        X = self.subsample(self.X, sample_size)
        X1 = self.subsample(self.X, sample_size)
        Y = self.subsample(self.Y, sample_size)
        X, Y = self.create_monotonic_dataset(X, X1, Y, distubance_level)
        X_train_dl = DataLoader(TensorDataset(X[:-num_cut]), batch_size=128)
        Y_train_dl = DataLoader(TensorDataset(Y[:-num_cut]), batch_size=128)

        X_test_dl = DataLoader(TensorDataset(X[-num_cut:]), batch_size=128)
        Y_test_dl = DataLoader(TensorDataset(Y[-num_cut:]), batch_size=128)

        metrics = {}
        metrics.update({'disturbance':distubance_level})

        sig_mmd = to_numpy(Sig_mmd(X, Y, depth=4))
        metrics.update({'sig_mmd':sig_mmd})

        d_score_mean, _ = compute_discriminative_score(
            X_train_dl, X_test_dl, Y_train_dl, Y_test_dl, self.config, self.config.dscore_hidden_size,
            num_layers=self.config.dscore_num_layers, epochs=self.config.dscore_epochs, batch_size=128)
        metrics.update({'discriminative score':d_score_mean})

        p_score_mean, _ = compute_predictive_score(
            X_train_dl, X_test_dl, Y_train_dl, Y_test_dl, self.config, self.config.pscore_hidden_size,
            self.config.pscore_num_layers, epochs=self.config.pscore_epochs, batch_size=128)
        metrics.update({'predictive scores': p_score_mean})

        fid_model = sig_fid_model(X, self.config)
        sig_fid = Predictive_FID(X, model=fid_model, name='Predictive_FID')(Y)
        metrics.update({'signature fid': to_numpy(sig_fid)})

        sig_kid = Predictive_KID(X, model=fid_model, name='Predictive_KID')(Y)
        metrics.update({'signature kid': to_numpy(sig_kid)})
        return metrics

    

    def run_montontic_test(self, num_run: int, distubance_level: int, sample_size):
        d_scores = []
        p_scores = []
        Sig_MMDs = []
        disturbance = []
        sig_fids = []
        sig_kids = []
        for i in tqdm(range(distubance_level+1)):
            for j in range(num_run):
                m = self.run_montontic_test_per_level(i, sample_size)
                d_scores.append(m['discriminative score'])
                p_scores.append(m['predictive scores'])
                Sig_MMDs.append(m['sig_mmd'])
                disturbance.append(m['disturbance'])
                sig_fids.append(m['signature fid'])
                sig_kids.append(m['signature kid'])
                disturbance.append(i)
        return pd.DataFrame({'sig_mmd': Sig_MMDs, 'signature fid': sig_fids, 'signature kid': sig_kids, 'predictive scores': p_scores, 'discriminative score': d_scores, 'disturbance': disturbance})

    def permutation_test(self, test_func, num_perm, sample_size):
        with torch.no_grad():
            X = self.subsample(self.X, sample_size)
            Y = self.subsample(self.Y, sample_size)
            X = X.to(self.config.device)
            Y = Y.to(self.config.device)

            # print(t1)
            n, m = X.shape[0], Y.shape[0]
            combined = torch.cat([X, Y])
            H0_stats = []
            H1_stats = []

            for i in range(num_perm):
                idx = torch.randperm(n+m)
                H0_stats.append(
                    test_func(combined[idx[:n]], combined[idx[n:]]).cpu().detach().numpy())
                H1_stats.append(test_func(self.subsample(
                    self.X, sample_size).to(self.config.device), self.subsample(self.Y, sample_size).to(self.config.device)).cpu().detach().numpy())
            Q_a = np.quantile(np.array(H0_stats), q=0.95)
            Q_b = np.quantile(np.array(H1_stats), q=0.05)

            # print(statistics)
            # print(np.array(statistics))
            power = 1 - (Q_a > np.array(H1_stats)).sum()/num_perm
            type1_error = (Q_b < np.array(H0_stats)).sum()/num_perm
        return power, type1_error


if __name__ == '__main__':
    import ml_collections
    import yaml
    import os
    config_dir = 'configs/' + 'test_metrics.yaml'
    with open(config_dir) as file:
        config = ml_collections.ConfigDict(yaml.safe_load(file))
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_id
    print(os.environ["CUDA_VISIBLE_DEVICES"])
    if config.dataset == 'AR':
        X = torch.FloatTensor(
            AROne(D=5, T=30, phi=np.linspace(0.1, 0.5, 5), s=0.5).batch(50000))
        Y = torch.FloatTensor(
            AROne(D=5, T=30, phi=np.linspace(0.6, 1, 5), s=0.5).batch(50000))
    if config.dataset == 'GBM':
        X = get_gbm(50000, 30, 5, drift=0.02, scale=0.1)
        Y = get_gbm(50000, 30, 5, drift=0.04, scale=0.1)
    df = Compare_test_metrics(X, Y, config).run_montontic_test(
        num_run=3, distubance_level=4, sample_size=10000)
    df.to_csv('numerical_results/test_metrics_test_' +
              config.dataset + '_complexnet.csv')
