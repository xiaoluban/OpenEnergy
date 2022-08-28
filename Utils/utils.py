import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.neighbors import KernelDensity
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal, Normal
from torch.distributions.distribution import Distribution

def r_2(y, yhat):
    SS_Residual = sum((y - yhat) ** 2)
    SS_Total = sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (float(SS_Residual)) / SS_Total
    # adjusted_r_squared = 1 - (1 - r_squared) * (len(y) - 1) / (len(y) - y.shape[1] - 1)
    return r_squared

def metrics(output1, gt1):
    # output1 = np.concatenate((output1[:, 0:1], output1[:, 11:]), 1)
    # gt1 = np.concatenate((gt1[:, 0:1], gt1[:, 11:]), 1)
    gt1 = gt1.astype(np.int)
    output1 = output1.astype(np.int)

    gt = gt1[gt1 != 0]
    output = output1[gt1 != 0]

    mae = np.fabs(output - gt).mean()
    t = output - gt
    mse = ((output - gt)**2).mean()
    rmse = np.sqrt(((output - gt)*(output - gt)).mean())
    # t= (output - gt)**2
    # t1 = ((output - gt)**2).mean()

    mape = np.abs((output - gt)/gt)

    mape = mape.mean()
    if mape > 1.0:
        ttt =0
    # print(mape)
    # print(mape)
    # mape = mape.mean()

    # r2 = r_2(gt, output)

    r2 = r2_score(gt, output)
    # print(output, gt)

    return mse, rmse, mape,r2, mae



def cut_egdes(num_grids, data_out, w_out):
    data = torch.stack(data_out, 2)
    weights = torch.stack(w_out, 2)

    att_w, att_j = torch.topk(weights, num_grids, 2)
    att_m = torch.zeros(att_j.size()[0], att_j.size()[1], data.size()[2] - att_j.size()[2]).type(torch.LongTensor)
    att_y = torch.cat((att_j, att_m), 2)
    att_y = weights - weights.detach() + att_y
    att_y = att_y.type(torch.LongTensor)

    data_out = torch.zeros(att_j.size())
    for i in range(data_out.size()[0]):
        for j in range(data_out.size()[1]):
            data_out[i, j, :] = data[i, j, att_y[i, j, :num_grids]]

    data_out1 = []
    weights_out1 = []
    for i_out in range(data_out.size()[2]):
        data_out1.append(data_out[:,:, i_out])
        weights_out1.append(att_j[:, :, i_out])

    return data_out1, weights_out1
def plot_all(output, gt):
    times = np.arange(0, 24, 1)
def cal_loss(gt, output, huberloss, loss_kl):


    loss = 1*torch.mean(torch.sqrt(torch.mul(output - gt, output - gt)))#1*torch.mean(torch.abs(output - gt)) +


    return loss

class GaussianKDE(Distribution):
    def __init__(self, X, bw):
        """
        X : tensor (n, d)
          `n` points with `d` dimensions to which KDE will be fit
        bw : numeric
          bandwidth for Gaussian kernel
        """
        self.X = X
        self.bw = bw
        self.dims = X.shape[-1]
        self.n = X.shape[0]
        self.mvn = MultivariateNormal(loc=torch.zeros(self.dims),
                                      covariance_matrix=torch.eye(self.dims))

    def sample(self, num_samples):
        idxs = (np.random.uniform(0, 1, num_samples) * self.n).astype(int)
        norm = Normal(loc=self.X[idxs], scale=self.bw)
        return norm.sample()

    def score_samples(self, Y, X=None):
        """Returns the kernel density estimates of each point in `Y`.

        Parameters
        ----------
        Y : tensor (m, d)
          `m` points with `d` dimensions for which the probability density will
          be calculated
        X : tensor (n, d), optional
          `n` points with `d` dimensions to which KDE will be fit. Provided to
          allow batch calculations in `log_prob`. By default, `X` is None and
          all points used to initialize KernelDensityEstimator are included.


        Returns
        -------
        log_probs : tensor (m)
          log probability densities for each of the queried points in `Y`
        """
        if X == None:
            X = self.X
        log_probs = torch.log(
            (self.bw**(-self.dims) *
             torch.exp(self.mvn.log_prob(
                 (X.unsqueeze(1) - Y) / self.bw))).sum(dim=0) / self.n)

        return log_probs

    def log_prob(self, Y):
        """Returns the total log probability of one or more points, `Y`, using
        a Multivariate Normal kernel fit to `X` and scaled using `bw`.

        Parameters
        ----------
        Y : tensor (m, d)
          `m` points with `d` dimensions for which the probability density will
          be calculated

        Returns
        -------
        log_prob : numeric
          total log probability density for the queried points, `Y`
        """

        X_chunks = self.X.split(1000)
        Y_chunks = Y.split(1000)

        log_prob = 0

        for x in X_chunks:
            for y in Y_chunks:
                log_prob += self.score_samples(y, x).sum(dim=0)

        return log_prob

def loss_density(rto, bias, rto1, bias1,  input, gt):

    rto = rto[:,:, 0, 0].reshape(rto.size()[0]*rto.size()[1], 1)
    bias = bias[:, :, 0, 0].reshape(bias.size()[0] * bias.size()[1], 1)
    bias1 = bias1[:, :, 0, 0].reshape(bias1.size()[0] * bias1.size()[1], 1)

    gt_tem = gt.reshape(gt.size()[0] * gt.size()[1], 1) - input.reshape(input.size()[0] * input.size()[1], 1) / rto
    prob = (1/bias1) * torch.exp(-(gt_tem-bias)*(gt_tem-bias)/(2*bias1*bias1))
    prob = -torch.log(prob) * 1

def loss_rto(rto, bias, input, gt):

    rto = rto[:,:, 0, 0] #.reshape(rto.size()[0]*rto.size()[1], 1)
    bias = bias[:, :, 0, 0]#.reshape(bias.size()[0] * bias.size()[1], 1)
    loss_kl = nn.KLDivLoss(reduction='mean')
    # loss_mse = nn.MSELoss(reduction='mean')

    pred = input/rto #+ bias

    loss1 = torch.sqrt(torch.mean((pred - gt) * (pred - gt)))
    # loss1 = loss_mse(pred, gt)
    loss2 = loss_kl((pred.softmax(-1) + torch.exp(torch.tensor(-10.0))).log(),
                        gt.softmax(-1) + torch.exp(torch.tensor(-10.0))) \
                + loss_kl((gt.softmax(-1) + torch.exp(torch.tensor(-10.0))).log(),
                          pred.softmax(-1) + torch.exp(torch.tensor(-10.0)))

    return loss1 #+ loss2 * 10

def loss_density_mdn(rto_tal, bias_tal, pi_tal, gt):
    prob_final = 0
    for i_gau in range(pi_tal.size()[2]):
        rto = rto_tal[:, :, i_gau, 0].reshape(rto_tal.size()[0] * rto_tal.size()[1], 1)
        bias = bias_tal[:, :, i_gau, 0].reshape(bias_tal.size()[0] * bias_tal.size()[1], 1)
        pi = pi_tal[:, :, i_gau, 0].reshape(pi_tal.size()[0] * pi_tal.size()[1], 1)

        gt_tem = gt.reshape(gt.size()[0] * gt.size()[1], 1)
        prob = (1 / bias) * torch.exp(-(gt_tem - rto) * (gt_tem - rto) / (2 * bias * bias))
        prob = -torch.log(prob + 0.00000000001) * pi
        prob = prob.reshape(gt.size()[0], gt.size()[1])
        prob_mean = torch.mean(prob, 0)
        prob_final += torch.mean(prob_mean)
        # print(pi)


    return prob_final#torch.max(prob_final)


def loss_density_pi(rto_tal, bias_tal, bias1_tal, pi_tal, input, gt):
    prob_final = 0
    for i_gau in range(pi_tal.size()[2]):
        rto = rto_tal[:, :, i_gau, 0].reshape(rto_tal.size()[0] * rto_tal.size()[1], 1)
        bias = bias_tal[:, :, i_gau, 0].reshape(bias_tal.size()[0] * bias_tal.size()[1], 1)
        bias1 = bias1_tal[:, :, i_gau, 0].reshape(bias1_tal.size()[0] * bias1_tal.size()[1], 1)
        pi = pi_tal[:, :, i_gau, 0].reshape(pi_tal.size()[0] * pi_tal.size()[1], 1)

        gt_tem = gt.reshape(gt.size()[0] * gt.size()[1], 1) - input.reshape(input.size()[0] * input.size()[1], 1) / rto
        prob = (1 / bias1) * torch.exp(-(gt_tem - bias) * (gt_tem - bias) / (2 * bias1 * bias1))
        prob = -torch.log(prob + 0.00000000001) * pi
        prob = prob.reshape(gt.size()[0], gt.size()[1])
        prob_mean = torch.mean(prob, 0)
        prob_final += torch.mean(prob_mean)
        # print(pi)


    return prob_final#torch.max(prob_final)


# def loss_density(rto, bias, rto1, bias1, input, gt):
#
#     rto = rto[:,:, 0, 0].reshape(rto.size()[0]*rto.size()[1], 1)
#     rto1 = rto1[:, :, 0, 0].reshape(rto1.size()[0] * rto1.size()[1], 1)
#     bias = bias[:, :, 0, 0].reshape(bias.size()[0] * bias.size()[1], 1)
#     bias1 = bias1[:, :, 0, 0].reshape(bias1.size()[0] * bias1.size()[1], 1)
#
#     samples_tem = input.reshape(input.size()[0] * input.size()[1], 1) / (1/(rto1)*torch.exp(-rto*rto/(rto1*rto1))) \
#                   + (1/(bias1)*torch.exp(-bias*bias/(bias1*bias1)))
#
#     normal_rto = torch.distributions.Normal(rto, rto1)
#     normal_bias = torch.distributions.Normal(bias, bias1)
#     samples_rto = []
#     samples_bias = []
#     for _ in range(10000):
#         samples_rto.append(normal_rto.sample())
#         samples_bias.append(normal_bias.sample())
#     samples_rto = torch.stack(samples_rto)
#     samples_bias = torch.stack(samples_bias)
#
#     loss = []
#     for i in range(rto.size()[0]):
#
#         samples_rto1 = samples_rto[:, i, 0:1]
#         samples_bias1 = samples_bias[:, i, 0:1]
#         t = input.reshape(input.size()[0] * input.size()[1], 1)[i:i+1, 0:1]
#         samples =  t/samples_rto1 + samples_bias1
#
#
#
#         # kde = GaussianKDE(samples, bw=0.1)
#         # kde.sample(10000)
#         # logprob = kde.score_samples(gt.reshape(gt.size()[0] * gt.size()[1], 1)[i:i+1, :])
#
#         bw = torch.tensor(0.1)
#         gt_tem = gt.reshape(gt.size()[0] * gt.size()[1], 1)[i:i+1, :]
#         count = 0
#         upbound = gt_tem + bw
#         lowerbound = gt_tem - bw
#         idx1 = samples > lowerbound
#         idx2 = samples < upbound
#         idx3 = idx1 * idx2
#         count = torch.sum(idx3)
#
#         # for j in range(samples.size()[0]):
#         #     jj = samples[j]
#         #     if gt_tem[0, 0]+bw > jj[0] and gt_tem[0, 0]-bw < jj[0]:
#         #         count += 1
#         if count == 0:
#             count = 1
#         #print(count)
#         logprob = -torch.log(torch.tensor(count/10000.0))
#
#
#
#         # kde = KernelDensity(bandwidth=0.1, kernel='gaussian')
#         # kde.fit(samples.detach().numpy())
#         # logprob = kde.score_samples(gt.reshape(gt.size()[0] * gt.size()[1], 1)[i:i+1, :])
#         # logprob = torch.unsqueeze(torch.from_numpy(logprob), 1)
#         #logprob = samples_tem - samples_tem.detach() + logprob
#         # print(logprob)
#
#         # logprob.requires_grad = True
#         # logprob = t - t.detach() + logprob
#         loss.append(logprob)
#
#     loss = torch.stack(loss)
#     loss = torch.unsqueeze(loss, 1)
#     loss = samples_tem - samples_tem.detach() + loss
#     # loss = rto - rto +  bias - bias + rto1 - rto1+ bias1 - bias1 + loss
#     # loss = 1/rto - 1/rto.detach() + 1/rto1 - 1/rto1.detach() + bias - bias.detach() + bias1 - bias1.detach() + loss
#
#
#     # samples_rto = torch.normal(rto, rto1)
#     # samples_bias = torch.normal(bias, bias1)
#     # samples = input.reshape(input.size()[0]*input.size()[1], 1)/samples_rto + samples_bias
#     #
#     # kde = KernelDensity(bandwidth=0.001, kernel='gaussian')
#     # kde.fit(samples.detach().numpy())
#     # logprob = kde.score_samples(gt.reshape(samples.size()[0]*samples.size()[1], 1))
#     # logprob = torch.from_numpy(logprob)
#     # logprob = torch.unsqueeze(logprob, 1)
#     # logprob = samples - samples.detach() + logprob
#     return torch.mean(loss)


