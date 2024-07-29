import torch

from networks.layers import *
from scipy import optimize
from multiprocessing import Pool
from networks.pose_net import *
import torch.optim as optim
import pytorch3d as p3d


class CuboidFitNN:
    def __init__(self, lr=1e-5, a_max=2., a_min=0.01, device=None, load=None, layers=4, train=True, arch="transformer",
                 refine=False, num_hull_samples=32):

        self.refine = refine

        print("Solver architecture: ", arch)

        if arch == 'transformer':
            self.network = TransformerPoseNet(layers, 3, a_min=a_min, a_max=a_max)
        else:
            assert False

        if load is not None:
            loaded_dict = torch.load(load, map_location=device)
            self.network.load_state_dict(loaded_dict)
        self.network.to(device)

        self.device = device
        self.train_solver = train
        if self.train_solver:
            self.optimiser = optim.Adam(self.network.parameters(), lr=lr, eps=1e-4, weight_decay=1e-4)
        self.a_max = a_max
        self.a_min = a_min

        self.num_hull_samples = num_hull_samples

    def backward_step(self):
        if self.train_solver:
            self.optimiser.step()

    def zero_grad(self):
        if self.train_solver:
            self.optimiser.zero_grad()

    def save_checkpoint(self, path, epoch, is_best=False):
        torch.save(self.network.state_dict(), '%s/primitive_fit_weights_%06d.net' % (path, epoch))
        if self.train_solver:
            torch.save(self.optimiser.state_dict(), '%s/primitive_fit_optimizer_%06d.net' % (path, epoch))
        if is_best:
            torch.save(self.network.state_dict(), '%s/primitive_fit_weights_best.net' % (path))
            if self.train_solver:
                torch.save(self.optimiser.state_dict(), '%s/primitive_fit_optimizer_best.net' % (path))

    def load_checkpoint(self, path, epoch):
        loaded_dict = torch.load('%s/primitive_fit_weights_%06d.net' % (path, epoch), map_location=self.device)
        self.network.load_state_dict(loaded_dict)
        if self.train_solver:
            loaded_dict = torch.load('%s/primitive_fit_optimizer_%06d.net' % (path, epoch))
            self.optimiser.load_state_dict(loaded_dict)

    def fit(self, data, choice, mask=None, **kwargs):

        B = data.size(0)
        Y = data.size(1)
        C = data.size(2)

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
            del choice_, data_, choice

        N = selected_data.size(4)

        selected_data_batched = selected_data.view(-1, N, C).to(self.device)

        if mask is not None:
            weights = mask.view(-1, N).float()

            choice_batched = torch.multinomial(weights, self.num_hull_samples, replacement=True)
            choice_batched = choice_batched.view(choice_batched.size(0), choice_batched.size(1), 1).\
                expand(choice_batched.size(0), choice_batched.size(1), 3)
            # weighted = True
            # selected_data_batched_w = torch.cat([selected_data_batched, weights.unsqueeze(-1)], dim=-1)

            selected_data_batched_w = torch.gather(selected_data_batched, 1, choice_batched)

            # means = torch.sum(selected_data_batched*weights.unsqueeze(-1), -2) / torch.sum(weights, -1).unsqueeze(-1)



        else:
            # weighted = False
            # means = torch.mean(selected_data_batched, -2)
            selected_data_batched_w = selected_data_batched

        means = torch.mean(selected_data_batched_w, -2)

        cnn_input = selected_data_batched_w-means.unsqueeze(1)
        cnn_out = self.network(cnn_input / (self.a_max/2.0))

        if len(cnn_out) == 4:
            size, axis, angle, position = cnn_out
            axis_angle = axis * angle
        else:
            size, axis_angle, position = cnn_out

        size = size * (self.a_max-self.a_min) + self.a_min
        position = self.a_max * position + means

        optim_params_batched = torch.cat([size, axis_angle, position], dim=-1)

        # residuals = cuboid_distance_batched(optim_params_batched.detach(), selected_data_batched.detach())
        residuals = cuboid_distance_batched(optim_params_batched, selected_data_batched, a_min=self.a_min, a_max=self.a_max)
        mean_residual = torch.mean(residuals)
        # mean_residual = torch.mean(residuals/((self.a_max/2.0)**2))
        optim_params = optim_params_batched.view(P, S, K, B, 9)

        if not self.train_solver:
            optim_params = optim_params.detach()

        del optim_params_batched
        del means
        del selected_data
        del selected_data_batched

        return optim_params, mean_residual, residuals

    def implicit_fun(self, params, points):
        return params

    def train(self):
        self.network.train()

    def eval(self):
        self.network.eval()

from itertools import combinations
def estimate_planes(points, threshold=0.05):

    Y = points.size(1)

    comb = torch.from_numpy(np.array(list(combinations(range(Y), 3)))).to(points.device)
    comb = comb.unsqueeze(0).unsqueeze(-1).expand(-1, -1, -1, 4)

    hom_points = torch.cat([points, torch.ones((points.size(0), points.size(1), 1))], -1)

    hom_points_ = hom_points.unsqueeze(1).expand(-1, comb.size(1), -1, -1)

    selection = torch.gather(hom_points_, 2, comb)
    means = torch.mean(selection, dim=2).unsqueeze(2)
    selection = torch.cat([selection, means], dim=2)

    u, s, vh = torch.svd(selection.cpu())

    planes = vh[:, :, :, -1]
    plane_norms = torch.norm(planes[:, :, 0:3], dim=-1).unsqueeze(-1)
    planes = planes / plane_norms

    distances = torch.abs(planes @ hom_points.transpose(-1, -2))

    inliers = (distances < threshold).sum(-1)

    best_indices = torch.argsort(inliers, dim=-1, descending=True)

    best_planes = torch.gather(planes, 1, best_indices[:, :3].unsqueeze(-1).expand(-1, -1, 4))

    return best_planes


def cuboid_single_plane(planes, points):
    B_ = planes.size(0)

    hom_points = torch.cat([points, torch.ones((points.size(0), points.size(1), 1))], -1)
    best_plane = planes[:, 0, :]

    normals = best_plane[:, :3]
    translation = -best_plane[:, 3] * normals
    translation = torch.mean(points, dim=1)

    one_vec = torch.tensor([0, 0, 1.], device=points.device).view(1, 3)

    u1 = torch.cross(normals, one_vec, dim=-1)
    u2 = torch.cross(normals, u1, dim=-1)

    u1 = u1 / torch.norm(u1, dim=-1, keepdim=True)
    u2 = u2 / torch.norm(u2, dim=-1, keepdim=True)

    R = torch.cat([u1.unsqueeze(1), u2.unsqueeze(1), normals.unsqueeze(1)], dim=1)

    points_t = points - translation.unsqueeze(1)
    points_t = points_t @ R.transpose(-1, -2)

    max_coords = torch.max(torch.abs(points_t), 1)[0]
    widths = max_coords.detach()

    R *= torch.det(R)[:, None, None]
    q_ = p3d.transforms.matrix_to_quaternion(R)
    q = p3d.transforms.quaternion_to_axis_angle(q_).detach()

    init_params_ = torch.tensor([[widths[i, 0], widths[i, 1], widths[i, 2],
                                  q[i, 0], q[i, 1], q[i, 2],
                                  translation[i, 0], translation[i, 1], translation[i, 2]] for i in range(B_)],
                                dtype=torch.float32, requires_grad=True, device=points.device)

    return init_params_


def cuboid_three_planes(planes, points):
    B_ = planes.size(0)

    best_plane = planes[:, 0, :]

    normals = planes[:, :, :3]
    translation = torch.mean(points, dim=1)

    R = torch.cat([normals[:, 0].unsqueeze(1), normals[:, 1].unsqueeze(1), normals[:, 2].unsqueeze(1)], dim=1)

    points_t = points - translation.unsqueeze(1)
    points_t = points_t @ R.transpose(-1, -2)

    max_coords = torch.max(torch.abs(points_t), 1)[0]
    widths = max_coords.detach()

    R *= torch.det(R)[:, None, None]
    q_ = p3d.transforms.matrix_to_quaternion(R)
    q = p3d.transforms.quaternion_to_axis_angle(q_).detach()

    init_params_ = torch.tensor([[widths[i, 0], widths[i, 1], widths[i, 2],
                                  q[i, 0], q[i, 1], q[i, 2],
                                  translation[i, 0], translation[i, 1], translation[i, 2]] for i in range(B_)],
                                dtype=torch.float32, requires_grad=True, device=points.device)

    return init_params_


class CuboidFitAdam:

    def __init__(self, iterations=100, lr=0.2, verbose=False, max_loss=1e-4, a_max=2., a_min=0.01, norm_by_volume=True,
                 device=None, sigmoid_size=False):

        self.iterations = iterations
        self.lr = lr
        self.verbose = verbose
        self.max_loss = max_loss
        self.a_max = a_max
        self.a_min = a_min
        self.norm_by_volume = norm_by_volume
        self.backward_object = CuboidWithImplicitFunGradient()
        self.device = device
        self.sigmoid_size = sigmoid_size

    def train(self):
        return

    def backward_step(self):
        return

    def zero_grad(self):
        return

    def save_checkpoint(self, path, epoch, is_best=False):
        return

    def fit(self, data, choice, mask=None, **kwargs):

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

        if mask is not None:
            weights = mask.view(-1, N).float()
            weighted = True
            selected_data_batched_w = torch.cat([selected_data_batched, weights.unsqueeze(-1)], dim=-1)

            means = torch.sum(selected_data_batched*weights.unsqueeze(-1), -2) / torch.sum(weights, -1).unsqueeze(-1)

        else:
            weighted = False
            means = torch.mean(selected_data_batched, -2)
            selected_data_batched_w = selected_data_batched

        B_ = selected_data_batched.size(0)
        means_ = means.detach()

        centred_points = selected_data_batched - means.unsqueeze(1)


        if mask is None:
            u, s, v = torch.pca_lowrank(centred_points.cpu())
            centred_rotated_points = centred_points @ v.to(data.device)
            R = v.transpose(-1,-2).contiguous()

            max_coords = torch.max(centred_rotated_points, 1)[0]
            min_coords = torch.min(centred_rotated_points, 1)[0]
            offsets = (max_coords + min_coords) / 2.
            centred_rotated_points = centred_rotated_points - offsets.unsqueeze(1)

            max_coords = torch.max(torch.abs(centred_rotated_points), 1)[0]
            if self.sigmoid_size:
                p = torch.clamp((max_coords-self.a_min)/(self.a_max-self.a_min), 1e-6, 1-1e-6)
                widths_ = (torch.log(p/(1-p))).detach()
            else:
                widths_ = max_coords.detach()
        else:
            centred_rotated_points = torch.clone(centred_points)
            R = torch.zeros(B_, 3, 3).to(centred_rotated_points.device)
            widths_ = torch.zeros(B_, 3).to(centred_rotated_points.device)
            offsets = torch.zeros(B_, 3).to(centred_rotated_points.device)
            for bi in range(B_):
                try:
                    sel_points = centred_points[bi, torch.nonzero(weights[bi]).squeeze(-1), :]
                    u, s, v = torch.pca_lowrank(sel_points.cpu())
                    sel_points = sel_points @ v.to(data.device)
                    max_coords = torch.max(sel_points, 0)[0]
                    min_coords = torch.min(sel_points, 0)[0]
                    offsets[bi] = (max_coords + min_coords) / 2.
                    sel_points = sel_points - offsets[bi].unsqueeze(0)
                    centred_rotated_points[bi, torch.nonzero(weights[bi]).squeeze(-1)] = sel_points

                    max_coords = torch.max(torch.abs(sel_points), 0)[0]
                    if self.sigmoid_size:
                        p = torch.clamp((max_coords-self.a_min)/(self.a_max-self.a_min), 1e-6, 1-1e-6)
                        widths_[bi] = (torch.log(p/(1-p))).detach()
                    else:
                        widths_[bi] = max_coords.detach()
                    R[bi] = v.transpose(-1,-2).contiguous() #here
                except:
                    R[bi] = torch.eye(3).to(R.device)

        R *= torch.det(R)[:, None, None]

        means_ = means_ + (offsets.unsqueeze(1) @ v.to(data.device)).squeeze(1)

        q_ = p3d.transforms.matrix_to_quaternion(R)
        q = p3d.transforms.quaternion_to_axis_angle(q_).detach()
        init_params_ = torch.tensor([[widths_[i, 0], widths_[i, 1], widths_[i, 2],
                                      q[i,0], q[i,1], q[i,2],
                                      means_[i,0], means_[i,1], means_[i,2]] for i in range(B_)],
                                      dtype=torch.float32, requires_grad=True, device=data.device)

        grads_enabled = torch.is_grad_enabled()
        torch.set_grad_enabled(True)
        init_params = init_params_.detach().requires_grad_(True)

        optim_params_batched, loss = \
            cuboid_from_points_weighted_torch_adam(selected_data_batched_w, init_params, weighted=weighted, a_max=self.a_max,
                                                   lr=self.lr, a_min=self.a_min, norm_by_volume=self.norm_by_volume,
                                                   iterations=self.iterations, sigmoid_size=self.sigmoid_size, **kwargs)
        torch.set_grad_enabled(grads_enabled)
        del loss

        if self.sigmoid_size:
            optim_params_batched[:, 0:3] = torch.sigmoid(optim_params_batched[:, 0:3]) * (self.a_max-self.a_min) + self.a_min
        else:
            optim_params_batched[:, 0:3] = torch.clamp(optim_params_batched[:, 0:3], min=self.a_min, max=self.a_max)

        optim_params_batched = self.implicit_fun(optim_params_batched.detach(), selected_data_batched)

        residuals = cuboid_distance_batched(optim_params_batched, selected_data_batched, a_min=self.a_min, a_max=self.a_max)

        # mean_residual = torch.mean(torch.sqrt(residuals))
        mean_residual = torch.mean(residuals)

        optim_params = optim_params_batched.view(P, S, K, B, 9)

        del init_params_
        del optim_params_batched
        del means
        del means_
        del selected_data
        del selected_data_batched

        return optim_params, mean_residual, residuals

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

        residuals = cuboid_distance_batched(optim_params_batched, selected_data_batched)
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
                                           lr=0.2, decrease_volume=False, mask=None, **kwargs):

    optimizer = torch.optim.Adam([params], lr=lr)

    best_params = params.clone()

    loss = cuboid_distance_batched(params, points, weighted=weighted, reduce=True, **kwargs)
    best_loss = torch.mean(loss)

    mean_loss = 0

    for iter in range(iterations):

        # optimizer.zero_grad()
        # loss0 = torch.sum(params[:, :3])
        # loss0.backward()
        # optimizer.step()


        # if verbose:
        #     print("%4d - loss0: %.4f" % (iter, loss0.item()), end="\n")

        optimizer.zero_grad()
        loss = cuboid_distance_batched(params, points, weighted=weighted, reduce=True, **kwargs)
        mean_loss = torch.mean(loss)

        if mean_loss < best_loss:
            best_params = params.clone()
            best_loss = mean_loss

        if mean_loss.item() < max_loss:
            break

        mean_loss.backward()
        optimizer.step()

        if verbose:
            print("%4d - loss1: %.4f" % (iter, mean_loss.item()), end="\n")
        # print("%4d - loss: %.4f" % (iter, mean_loss.item()), end="\n")

        # if iter % 20 == 0:
        #     for param_group in optimizer.param_groups:
        #         param_group['lr'] /= 2.

    del optimizer

    return best_params, mean_loss


def cuboid_from_points_constrained_adam(points, params, weighted=True, iterations=100, max_loss=1e-4, verbose=False,
                                           lr=0.2, angle_weight=0, optimizer=None, occlusion_aware=False, **kwargs):

    if optimizer is None:
        optimizer = torch.optim.Adam([params], lr=lr)

    best_params = params.clone()
    angle_constraint = cuboid_angle_constraint(params)
    if occlusion_aware:
        loss = cuboid_distance_occluded(params, points, weighted=weighted, reduce=True, **kwargs)
    else:
        loss = cuboid_distance_batched(params, points, weighted=weighted, reduce=True, **kwargs)
    best_loss = torch.mean(loss) + angle_constraint * angle_weight

    mean_loss = 0

    for iter in range(iterations):
        optimizer.zero_grad()
        if occlusion_aware:
            loss = cuboid_distance_occluded(params, points, weighted=weighted, reduce=True, **kwargs)
        else:
            loss = cuboid_distance_batched(params, points, weighted=weighted, reduce=True, **kwargs)
        angle_constraint = cuboid_angle_constraint(params)
        mean_loss = torch.mean(loss) + angle_constraint * angle_weight


        if mean_loss < best_loss:
            best_params = params.clone()
            best_loss = mean_loss

        if mean_loss.item() < max_loss:
            print("loss: ", mean_loss.item())
            break

        mean_loss.backward()
        optimizer.step()

        if verbose:
            print("%4d - loss: %.4f" % (iter, mean_loss.item()), end="\n")

        if iter % 20 == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] /= 2.

    # del optimizer

    return best_params, mean_loss, optimizer


def cuboid_from_points_weighted_torch_lbfgs(points, params, weighted=True, iterations=100, max_loss=1e-4, **kwargs):

    optimizer = torch.optim.LBFGS([params], lr=1e-1, max_iter=20, line_search_fn=None)

    def closure():
        optimizer.zero_grad()
        loss = cuboid_distance_batched(params, points, weighted=weighted, reduce=True, **kwargs)
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