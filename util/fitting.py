from networks.layers import *
from util.consistency import *
import torch
import kornia
from scipy import optimize
from multiprocessing import Pool


class CuboidFitAdam:

    def __init__(self, iterations=100, lr=0.2, verbose=False, max_loss=1e-4, a_max=2., a_min=0.01, norm_by_volume=True, device=None):

        self.iterations = iterations
        self.lr = lr
        self.verbose = verbose
        self.max_loss = max_loss
        self.a_max = a_max
        self.a_min = a_min
        self.norm_by_volume = norm_by_volume
        self.backward_object = CuboidWithImplicitFunGradient()
        self.device = device

    def train(self):
        return

    def backward_step(self):
        return

    def zero_grad(self):
        return

    def save_checkpoint(self, path, epoch, is_best=False):
        return

    def fit(self, data, choice, **kwargs):

        B = data.size(0)
        Y = data.size(1)
        C = data.size(2)

        data = data.detach()

        if choice is None:
            P, S, K = (1, 1, 1)
            selected_data = data.view(1, 1, 1, B, Y, C)
        else:
            P = choice.size(0)
            S = choice.size(1)
            K = choice.size(2)
            B = choice.size(3)
            N = choice.size(4)
            data_ = data.view(1, 1, 1, B, Y, C).expand(P, S, K, B, Y, C)
            choice = choice.detach()
            choice_ = choice.view(P, S, K, B, N, 1).expand(P, S, K, B, N, C)
            selected_data = torch.gather(data_, 4, choice_)
            del choice_, data_

        N = selected_data.size(4)

        selected_data_batched = selected_data.view(-1, N, C)

        B_ = selected_data_batched.size(0)
        means = torch.mean(selected_data_batched, -2)
        means_ = means.detach()

        centred_points = selected_data_batched - means.unsqueeze(1)

        u, s, v = torch.pca_lowrank(centred_points.cpu())
        centred_rotated_points = centred_points @ v.to(data.device)
        max_coords = torch.max(torch.abs(centred_rotated_points), 1)[0]
        widths_ = max_coords.detach()

        q = kornia.rotation_matrix_to_quaternion(v.transpose(-1,-2).contiguous())
        q_ = torch.zeros_like(q)
        q_[:, 0] = q[:, 3]
        q_[:, 1:4] = q[:, 0:3]
        q = kornia.quaternion_to_angle_axis(q_).detach()

        init_params_ = torch.tensor([[widths_[i, 0]/2., widths_[i, 1]/2., widths_[i, 2]/2.,
                                      q[i,0], q[i,1], q[i,2],
                                      means_[i,0], means_[i,1], means_[i,2]] for i in range(B_)],
                                      dtype=torch.float32, requires_grad=True, device=data.device)

        grads_enabled = torch.is_grad_enabled()
        torch.set_grad_enabled(True)
        init_params = init_params_.detach().requires_grad_(True)

        optim_params_batched, loss = \
            cuboid_from_points_weighted_torch_adam(selected_data_batched, init_params, weighted=False, a_max=self.a_max,
                                                   a_min=self.a_min, norm_by_volume=self.norm_by_volume, iterations=self.iterations, **kwargs)
        torch.set_grad_enabled(grads_enabled)
        del loss

        optim_params_batched[:, 0:3] = torch.clamp(optim_params_batched[:, 0:3], min=self.a_min, max=self.a_max)

        optim_params_batched = self.implicit_fun(optim_params_batched.detach(), selected_data_batched)

        residuals = cuboid_fun_weighted_torch(optim_params_batched, selected_data_batched)
        mean_residual = torch.mean(residuals)

        optim_params = optim_params_batched.view(P, S, K, B, 9)

        del init_params_
        del optim_params_batched
        del means
        del means_
        del selected_data
        del selected_data_batched

        return optim_params, mean_residual

    def implicit_fun(self, params, points):
        return self.backward_object.apply(params, points)


class CuboidFitLBFGS:

    def __init__(self, iterations=100, lr=0.2, verbose=False, max_loss=1e-4, a_max=2., a_min=0.01, norm_by_volume=True, device=None):

        self.iterations = iterations
        self.lr = lr
        self.verbose = verbose
        self.max_loss = max_loss
        self.a_max = a_max
        self.a_min = a_min
        self.norm_by_volume = norm_by_volume
        self.backward_object = CuboidWithImplicitFunGradient()
        self.device = device

    def backward_step(self):
        return

    def zero_grad(self):
        return

    def save_checkpoint(self, path, epoch, is_best=False):
        return

    def fit(self, data, choice, **kwargs):

        B = data.size(0)
        Y = data.size(1)
        C = data.size(2)

        data = data.detach()

        if choice is None:
            P, S, K = (1, 1, 1)
            selected_data = data.view(1, 1, 1, B, Y, C)
        else:
            P = choice.size(0)
            S = choice.size(1)
            K = choice.size(2)
            B = choice.size(3)
            N = choice.size(4)
            data_ = data.view(1, 1, 1, B, Y, C).expand(P, S, K, B, Y, C)
            choice = choice.detach()
            choice_ = choice.view(P, S, K, B, N, 1).expand(P, S, K, B, N, C)
            selected_data = torch.gather(data_, 4, choice_)
            del choice_, data_

        N = selected_data.size(4)

        selected_data_batched = selected_data.view(-1, N, C)

        B_ = selected_data_batched.size(0)
        means = torch.mean(selected_data_batched, -2)
        means_ = means.detach()

        centred_points = selected_data_batched - means.unsqueeze(1)

        u, s, v = torch.pca_lowrank(centred_points.cpu())
        centred_rotated_points = centred_points @ v.to(data.device)
        max_coords = torch.max(torch.abs(centred_rotated_points), 1)[0]
        widths_ = max_coords.detach()

        q = kornia.rotation_matrix_to_quaternion(v.transpose(-1,-2).contiguous())
        q_ = torch.zeros_like(q)
        q_[:, 0] = q[:, 3]
        q_[:, 1:4] = q[:, 0:3]
        q = kornia.quaternion_to_angle_axis(q_).detach()

        init_params_ = torch.tensor([[widths_[i, 0]/2., widths_[i, 1]/2., widths_[i, 2]/2.,
                                      q[i,0], q[i,1], q[i,2],
                                      means_[i,0], means_[i,1], means_[i,2]] for i in range(B_)],
                                      dtype=torch.float32, requires_grad=True, device=data.device)

        grads_enabled = torch.is_grad_enabled()
        torch.set_grad_enabled(True)
        init_params = init_params_.detach().requires_grad_(True)

        optim_params_batched, loss = \
            cuboid_from_points_numpy(selected_data_batched, init_params, a_max=self.a_max, a_min=self.a_min,
                                     norm_by_volume=self.norm_by_volume, **kwargs)
        torch.set_grad_enabled(grads_enabled)
        del loss

        optim_params_batched[:, 0:3] = torch.clamp(optim_params_batched[:, 0:3], min=self.a_min, max=self.a_max)

        residuals = cuboid_fun_weighted_torch(optim_params_batched, selected_data_batched)
        mean_residual = torch.mean(residuals)

        optim_params = optim_params_batched.view(P, S, K, B, 9)

        del init_params_
        del optim_params_batched
        del means
        del means_
        del selected_data
        del selected_data_batched

        return optim_params, mean_residual

    def implicit_fun(self, params, points):
        return self.backward_object.apply(params, points)


def cuboid_from_points_weighted_torch_adam(points, params, weighted=True, iterations=100, max_loss=1e-4, verbose=False,
                                           lr=0.2, **kwargs):

    optimizer = torch.optim.Adam([params], lr=lr)

    best_params = params.clone()

    loss = cuboid_fun_weighted_torch(params, points, weighted=weighted, reduce=True, **kwargs)
    best_loss = torch.mean(loss)

    mean_loss = 0

    for iter in range(iterations):
        optimizer.zero_grad()
        loss = cuboid_fun_weighted_torch(params, points, weighted=weighted, reduce=True, **kwargs)
        mean_loss = torch.mean(loss)

        if mean_loss < best_loss:
            best_params = params.clone()
            best_loss = mean_loss

        if mean_loss.item() < max_loss:
            break

        mean_loss.backward()
        optimizer.step()

        if verbose:
            print("%4d - loss: %.4f" % (iter, mean_loss.item()), end="\n")

        if iter % 20 == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] /= 2.

    del optimizer

    return best_params, mean_loss


def cuboid_from_points_weighted_torch_lbfgs(points, params, weighted=True, iterations=100, max_loss=1e-4, **kwargs):

    optimizer = torch.optim.LBFGS([params], lr=1e-1, max_iter=20, line_search_fn=None)

    def closure():
        optimizer.zero_grad()
        loss = cuboid_fun_weighted_torch(params, points, weighted=weighted, reduce=True, **kwargs)
        mean_loss = torch.mean(loss)

        mean_loss.backward()
        return mean_loss

    for iter in range(iterations):
        optimizer.step(closure)

    mean_loss = closure()

    del optimizer

    return params, mean_loss


def minimize_cuboid_from_points_numpy(args):
    a, x, verbose = args
    res = optimize.minimize(cuboid_fun_numpy, x, method='L-BFGS-B', jac=False, args=a, options={'disp': verbose, 'eps':1e-5})
    return res.x


def cuboid_from_points_numpy(points, params, verbose=False, norm_by_volume=False, **kwargs):

    B = points.size(0)

    points_np = points.detach().cpu().numpy()
    params_np = params.detach().cpu().numpy()

    args = [((points_np[i], norm_by_volume), params_np[i], verbose) for i in range(B)]
    p = Pool(20)

    results = p.map(minimize_cuboid_from_points_numpy, args)

    optim_params = torch.from_numpy(np.array(results, dtype=np.float32))

    p.close()
    p.join()

    return optim_params, 0