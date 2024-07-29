import numpy as np

from util import consistency
from util import fitting
import torch
import math


class CuboidEM:

    def __init__(self, initial_parameters, points, learning_rate, distance_function, sigma, verbose=False,
                 size_weight=0., angle_weight=0.):

        B, Y, C = points.size()
        P, M, K, B, D = initial_parameters.size()
        self.model_shape = (P, M, K, B, D)

        self.points = points.view(1, 1, 1, B, Y, C).expand(P, M, K, B, Y, C).view(-1, Y, C)
        self.cuboid_parameters = torch.clone(initial_parameters).view(-1, D).requires_grad_(True)
        self.priors = torch.ones(self.cuboid_parameters.size(0), device=self.cuboid_parameters.device).requires_grad_(True)

        self.sigmas = sigma * torch.ones(self.cuboid_parameters.size(0), device=initial_parameters.device, requires_grad=True)
        self.sigmas = self.sigmas.requires_grad_(True)
        self.optimizer = torch.optim.Adam([self.cuboid_parameters, self.sigmas, self.priors], lr=learning_rate)

        self.distance_function = distance_function
        self.verbose = verbose
        self.size_weight = size_weight
        self.angle_weight = angle_weight

    def log_likelihood(self):
        distances = self.distance_function(self.cuboid_parameters, self.points)
        if self.size_weight > 0:
            sizes = (self.cuboid_parameters[:,0] + self.cuboid_parameters[:,1] + self.cuboid_parameters[:,2])**2
            sizes = sizes[:, None]
            distances += sizes * self.size_weight

        exp_term = -distances / (2*self.sigmas[:, None]**2)
        log_term = -torch.log(torch.sqrt(2*math.pi*self.sigmas[:, None]**2))

        return exp_term + log_term


    def posterior(self, log_likelihood, priors=None):
        likelihood = torch.exp(log_likelihood)
        enumerator = likelihood * self.priors[:, None]
        denominator = torch.clamp(torch.sum(enumerator, dim=0), min=1e-8)
        posterior = enumerator/denominator[None, :]

        return posterior

    def q_function(self):
        log_lh = self.log_likelihood()
        posterior = self.posterior(log_lh)
        q_vals = log_lh * posterior
        q = q_vals.sum(-1).sum(-1)

        return q

    def step(self):
        self.optimizer.zero_grad()
        loss = -self.q_function()
        if self.angle_weight > 0:
            angle_loss = consistency.cuboid_angle_constraint(self.cuboid_parameters)
            print("angle loss: %.2f" % angle_loss.item())
            loss += self.angle_weight * angle_loss
        loss.backward()
        self.optimizer.step()
        return loss

    def closure(self):
        self.optimizer.zero_grad()
        loss = -self.q_function()
        loss.backward()
        return loss

    def run_iterations(self, iterations):

        grads_enabled = torch.is_grad_enabled()
        torch.set_grad_enabled(True)

        # first_loss = None
        # best_loss = np.Inf
        # best_params = torch.clone(self.cuboid_parameters).detach()
        for i in range(iterations):
            loss = self.step()
            # if first_loss is None:
            #     first_loss = loss.item()
            # loss = -self.q_function()

            # self.optimizer.step(self.closure)

            # loss = -self.q_function()
            # if loss.item() < best_loss:
            #     best_loss = loss.item()
            #     best_params = torch.clone(self.cuboid_parameters).detach()
            # else:
            #     self.cuboid_parameters = torch.clone(best_params).requires_grad_(True)

            # print(self.sigmas.cpu().detach().numpy(), self.priors.cpu().detach().numpy())

            # print("%4d - loss: %.4f" % (i, loss.item()), end="\n")

        # if self.verbose:
        #     print("Loss: %.2f -> %.2f" % (first_loss, loss.item()))

        torch.set_grad_enabled(grads_enabled)

        P, M, K, B, D = self.model_shape
        params = self.cuboid_parameters.view(P, M, K, B, D)#.transpose(1, -2)

        return params



def e_step(models, data, distance_fun, priors, variances, data_weights=None, **kwargs):
    # data: B x N x C
    # data_weights: B x N
    # models: B x M x D
    # priors: B x M
    # variances: B x M
    # distances: B x N x M

    B, N, C = data.size()
    B, M, D = models.size()

    data_ = data.unsqueeze(1).expand(-1, M, -1, -1).view(-1, N, C)
    models_ = models.view(-1, D)

    distances = distance_fun(models_, data_, **kwargs)

    distances = distances.view(B, M, N).transpose(-1, -2)
    distances = torch.cat([distances, torch.zeros((B, N, 1), device=distances.device)], dim=-1)

    if priors is None:
        priors = torch.ones((models.size(0), models.size(1)+1), device=models.device)
    if isinstance(variances, (int, float)):
        variances = variances * torch.ones((models.size(0), models.size(1)), device=models.device)
        variances_ = variances.unsqueeze(1).expand(-1, distances.size(1), -1)
    else:
        variances_ = variances.unsqueeze(1).expand(-1, distances.size(1), -1)

    variances_ = torch.cat([variances_, 1e-1 * torch.ones((B, N, 1), device=variances_.device)], dim=-1)

    # likelihood: B x N x M
    likelihood = 1. / torch.sqrt(2 * math.pi * variances_) * torch.exp(
        -distances / (2 * variances_)) # * data_weights_

    if data_weights is not None:
        data_weights_ = data_weights.unsqueeze(-1).expand(-1, -1, distances.size(2))
        likelihood = likelihood * data_weights_

    # marginal: B x N
    marginal = torch.max(torch.matmul(likelihood, priors.unsqueeze(-1)), 1e-8 * torch.ones(1, device=models.device))

    # posterior: B x N x M
    posterior = likelihood * priors.unsqueeze(-1).transpose(1, 2) / marginal

    posterior_sum = posterior.sum(-1).unsqueeze(-1)

    return (posterior / posterior_sum)[:,:, :-1]
    # return (posterior)[:,:, :-1]


def m_step_cuboids_adam(data, weights, models, adam_iter, optimizer=None, angle_weight=0.,
                        oa_estep=False, oa_mstep=False, a_min=0.01, a_max=3., max_var=1e-2):
    # data: B x N x C
    # weights: B x N x M


    B, N, C = data.size()
    B, M, D = models.size()

    weights_ = weights.transpose(0, -1)

    data_ = data.unsqueeze(1).expand(-1, M, -1, -1).view(-1, N, C)
    data_ = torch.cat([data_, weights_], dim=-1)
    models_ = models.view(-1, D)

    grads_enabled = torch.is_grad_enabled()
    torch.set_grad_enabled(True)
    models_ = models_.detach().requires_grad_(True)

    new_models, _, optimizer = fitting.cuboid_from_points_constrained_adam(data_, models_, weighted=True, iterations=adam_iter,
                                                                max_loss=1e-4, verbose=False, lr=1e-3, angle_weight=angle_weight,
                                                                norm_by_volume=True, occlusion_aware=oa_mstep)

    new_models[:, 0:3] = torch.clamp(new_models[:, 0:3], min=a_min, max=a_max)

    torch.set_grad_enabled(grads_enabled)

    if oa_estep:
        distances = consistency.cuboid_distance_occluded(models_, data_)
    else:
        distances = consistency.cuboid_distance_batched(new_models, data_, a_min=a_min, a_max=a_max)


    new_models = new_models.view(B, M, D)
    distances_ = distances.view(B, M, N).transpose(1, 2)

    distances_sq = distances_
    distances_sq_weighted = distances_sq * weights
    distances_sqwt_sum = distances_sq_weighted.sum(1)
    weights_sum = torch.max(weights.sum(1), 1e-8 * torch.ones(1, device=data.device))
    #
    variances = distances_sqwt_sum / weights_sum

    variances = torch.clip(variances, max=max_var)
    # variances = variances.sum(-1)
    # variances = None

    return new_models, variances, distances, optimizer


def em_for_cuboids(data, models, data_weights=None, init_variance=1e-4, iterations=1, device=None, adam_iter=10,
                   angle_weight=0, oa_estep=False, oa_mstep=False, max_var=1e-2):
    variances = init_variance
    optimizer = None
    all_models = [models.clone().detach()]
    all_posteriors = []
    for i in range(iterations):
        if oa_estep:
            posterior = e_step(models, data, consistency.cuboid_distance_occluded, None, variances, data_weights, a_min=0.001, a_max=100.0)
        else:
            posterior = e_step(models, data, consistency.cuboid_distance_batched, None, variances, data_weights, a_min=0.001, a_max=100.0)
        # posterior = e_step(models, data, consistency.cuboid_distance_occluded, None, variances, data_weights)
        models, variances, distances, optimizer = m_step_cuboids_adam(data, posterior, models, adam_iter,
                                                                      optimizer=optimizer, angle_weight=angle_weight,
                                                                      oa_mstep=oa_mstep, oa_estep=oa_estep,
                                                                      max_var=max_var)

        all_models += [models.clone().detach()]
        all_posteriors += [posterior.clone().detach()]
    all_posteriors += [posterior.clone().detach()]

    return models, posterior, variances, all_models, all_posteriors


def intermediate_refinement(data, models, **kwargs):

    B, Y, C = data.size()
    P, M, K, B, D = models.size()

    data_ = data.view(1, 1, B, Y, C).expand(P, K, B, Y, C).view(-1, Y, C)
    models_ = models.transpose(1, -2).view(-1, M, D)

    new_models_, _, _, all_models, all_posteriors = em_for_cuboids(data_, models_, **kwargs)

    new_models = new_models_.view(P, K, B, M, D).transpose(1, -2)

    return new_models, all_models, all_posteriors
