import torch
import numpy as np
from pyquaternion import Quaternion
import torchgeometry as tgm


def sigmoid(x):
    return 1. / (1. + torch.exp(-x))


def soft_inlier_fun_gen(beta, tau):
    def f(d):
        return 1 - sigmoid(beta * d - beta * tau)
    return f


def soft_outlier_fun_gen(beta, tau):
    def f(d):
        return sigmoid(beta * d - beta * tau)
    return f


def soft_inlier_fun(d, beta, tau):
    return 1 - sigmoid(beta * d - beta * tau)


def cuboids_consistency_measure_iofun_parallel_batched_torch(h, data, **kwargs):
    P = h.size(0)
    S = h.size(1)
    K = h.size(2)
    B = h.size(3)

    Y = data.size(1)
    C = data.size(2)

    h_batched = h.view(-1, 9)
    data_batched = data.view(1, 1, 1, B, Y, C)
    data_batched = data_batched.expand(P, S, K, B, Y, C)
    data_batched = data_batched.reshape(-1, Y, C)

    distances_batched = cuboid_fun_weighted_torch(h_batched, data_batched, reduce=False, weighted=False, **kwargs)

    distances = distances_batched.view(P, S, K, B, Y)

    del h_batched, data_batched, distances_batched

    return distances


def cuboid_fun_weighted_torch(params, points, reduce=False, weighted=False, a_max=2., a_min=0.01, radial_distance=False,
                              inside_only=False, norm_by_volume=False):

    params = torch.where(params != params, torch.zeros_like(params), params)

    B = params.size(0)
    Y = points.size(1)

    a1 = torch.clamp(params[:, 0], a_min, a_max).unsqueeze(-1)
    a2 = torch.clamp(params[:, 1], a_min, a_max).unsqueeze(-1)
    a3 = torch.clamp(params[:, 2], a_min, a_max).unsqueeze(-1)

    q = torch.clamp(params[:, 3:6], -100, 100)
    t = torch.clamp(params[:, 6:9], -100, 100)

    norms = torch.norm(points, dim=-1)

    if torch.isnan(q).any():
        assert False

    angle_axis = q
    R = tgm.angle_axis_to_rotation_matrix(angle_axis)
    R = R[:, :3, :3]

    t_ = t.unsqueeze(1)
    points_ = points[:, :, :3]
    points_transformed = points_ - t_
    points_transformed = torch.matmul(points_transformed, R.transpose(-2, -1))

    x = points_transformed[:, :, 0]
    y = points_transformed[:, :, 1]
    z = points_transformed[:, :, 2]

    zero = torch.FloatTensor([0]).to(x.device)

    outside_values = torch.max(torch.abs(x)-a1, zero)**2 + \
                     torch.max(torch.abs(y)-a2, zero)**2 + \
                     torch.max(torch.abs(z)-a3, zero)**2
    inside_values = torch.max(torch.min(a1-torch.abs(x),
                              torch.min(a2-torch.abs(y),
                                        a3-torch.abs(z))), zero)**2

    if inside_only:
        squared_distances = inside_values
    else:
        squared_distances = outside_values-inside_values

    squared_distances = torch.abs(squared_distances)

    if weighted:
        weights = points[:, :, 3]
        squared_distances *= weights

        squared_distances = torch.where(norms < 1e-6, torch.ones_like(squared_distances)*10000., squared_distances)

    if norm_by_volume:
        squared_distances = squared_distances * (a1 + a2 + a3)

    if reduce:
        return torch.sum(squared_distances, dim=-1)
    else:
        return squared_distances


def cuboid_fun_numpy(params, points, norm_by_volume, reduce=True, a_max=2., a_min=0.01):

    a1 = np.clip(params[0], a_min, a_max)
    a2 = np.clip(params[1], a_min, a_max)
    a3 = np.clip(params[2], a_min, a_max)

    q = np.clip(params[3:6], -100, 100)
    t = np.clip(params[6:9], -100, 100)

    points_ = points.copy()
    points_[:, 0] -= t[0]
    points_[:, 1] -= t[1]
    points_[:, 2] -= t[2]

    angle = np.linalg.norm(q)
    angle_axis = Quaternion(axis=q/angle, angle=angle)
    R = angle_axis.rotation_matrix

    points_transformed = points_ @ R.T

    x = points_transformed[:, 0]
    y = points_transformed[:, 1]
    z = points_transformed[:, 2]

    outside_values = np.maximum(np.abs(x)-a1, 0)**2 + \
                     np.maximum(np.abs(y)-a2, 0)**2 + \
                     np.maximum(np.abs(z)-a3, 0)**2
    inside_values = np.maximum(np.minimum(a1-np.abs(x),
                              np.minimum(a2-np.abs(y),
                                        a3-np.abs(z))), 0)**2

    squared_distances = outside_values-inside_values

    squared_distances = np.abs(squared_distances)

    if norm_by_volume:
        squared_distances = squared_distances * (a1 + a2 + a3)

    if reduce:
        return np.sum(squared_distances, axis=-1)
    else:
        return squared_distances


def cuboid_occlusion_torch(params, points, a_max=100., a_min=0.001):

    params = torch.where(params != params, torch.zeros_like(params), params)

    B = params.size(0)
    Y = points.size(1)

    a1 = torch.clamp(params[:, 0], a_min, a_max)
    a2 = torch.clamp(params[:, 1], a_min, a_max)
    a3 = torch.clamp(params[:, 2], a_min, a_max)
    q = params[:, 3:6]
    t = params[:, 6:9]

    angle_axis = q
    R = tgm.angle_axis_to_rotation_matrix(angle_axis)
    R = R[:, :3, :3]

    t_ = t.unsqueeze(1)
    points_ = points[:, :, :3]
    points_transformed = points_ - t_
    points_transformed = torch.matmul(points_transformed, R.transpose(-2, -1)) # BxYx3
    x = points_transformed[:, :, 0]
    y = points_transformed[:, :, 1]
    z = points_transformed[:, :, 2]

    origins = torch.zeros((B, 3)).to(params.device)
    origins_transformed_ = origins - t
    origins_transformed = torch.matmul(origins_transformed_.unsqueeze(1), R.transpose(-2, -1)) # Bx1x3

    support = points_transformed
    vectors = origins_transformed-points_transformed
    vector_norms = torch.norm(vectors, dim=-1, keepdim=True)
    vectors /= vector_norms

    plane_normals = torch.zeros((B, 6, 3)).to(params.device)
    plane_normals[:, 0:2, 0] = 1
    plane_normals[:, 2:4, 1] = 1
    plane_normals[:, 4:6, 2] = 1
    # plane_normals_ = plane_normals.unsqueeze(1) # Bx1x6x3
    plane_dists = torch.zeros((B, 6)).to(params.device)
    plane_dists[:, 0] = a1  # pos axis direction
    plane_dists[:, 1] = -a1 # neg axis direction
    plane_dists[:, 2] = a2
    plane_dists[:, 3] = -a2
    plane_dists[:, 4] = a3
    plane_dists[:, 5] = -a3
    plane_dists_ = plane_dists.unsqueeze(1) # Bx1x6

    lambdas = (plane_dists_ - torch.matmul(support, plane_normals.transpose(-1, -2))) \
              / torch.matmul(vectors, plane_normals.transpose(-1, -2)) # BxYx6

    lambdas_ = lambdas.unsqueeze(-1) # BxYx6x1
    support_ = support.unsqueeze(2)  # BxYx1x3
    vectors_ = vectors.unsqueeze(2)  # BxYx1x3

    intersections = support_ + lambdas_ * vectors_  # BxYx6x3

    a1_ = a1.unsqueeze(-1)
    a2_ = a2.unsqueeze(-1)
    a3_ = a3.unsqueeze(-1)  # Bx1
    a1__ = a1.unsqueeze(-1).unsqueeze(-1)
    a2__ = a2.unsqueeze(-1).unsqueeze(-1)
    a3__ = a3.unsqueeze(-1).unsqueeze(-1) # Bx1x1
    a_ = torch.stack([a1__, a2__, a3__], dim=-1) # Bx1x1x3


    zero = torch.FloatTensor([0]).to(x.device)
    distances_to_a1_pos = ((x-a1_)**2 + torch.max(torch.abs(y)-a2_, zero)**2 + torch.max(torch.abs(z)-a3_, zero)**2)
    distances_to_a1_neg = ((-x-a1_)**2 + torch.max(torch.abs(y)-a2_, zero)**2 + torch.max(torch.abs(z)-a3_, zero)**2)
    distances_to_a2_pos = ((y-a2_)**2 + torch.max(torch.abs(x)-a1_, zero)**2 + torch.max(torch.abs(z)-a3_, zero)**2)
    distances_to_a2_neg = ((-y-a2_)**2 + torch.max(torch.abs(x)-a1_, zero)**2 + torch.max(torch.abs(z)-a3_, zero)**2)
    distances_to_a3_pos = ((z-a3_)**2 + torch.max(torch.abs(x)-a1_, zero)**2 + torch.max(torch.abs(y)-a2_, zero)**2)
    distances_to_a3_neg = ((-z-a3_)**2 + torch.max(torch.abs(x)-a1_, zero)**2 + torch.max(torch.abs(y)-a2_, zero)**2)
    distances_to_cube_sides = torch.stack([distances_to_a1_pos, distances_to_a1_neg,
                                           distances_to_a2_pos, distances_to_a2_neg,
                                           distances_to_a3_pos, distances_to_a3_neg], dim=-1) # BxYx6

    dist_values = a_-torch.abs(intersections)

    intersections_on_cube = torch.where(dist_values >= -1e-2,
                                        torch.ones([1 for _ in range(dist_values.dim())]).to(dist_values.device),
                                        torch.zeros([1 for _ in range(dist_values.dim())]).to(dist_values.device))
    intersections_on_cube_ = intersections_on_cube[:, :, :, 0] *intersections_on_cube[:, :, :, 1]*intersections_on_cube[:, :, :, 2]

    possible_occlusion = torch.where(lambdas > 0,
                                     torch.ones([1 for _ in range(lambdas.dim())]).to(lambdas.device),
                                     torch.zeros([1 for _ in range(lambdas.dim())]).to(lambdas.device))

    occlusions = possible_occlusion * intersections_on_cube_

    a_ = torch.stack([a1_, a1_, a2_, a2_, a3_, a3_], dim=-1) # Bx1x6
    points_restacked = torch.stack([-points_transformed[:,:,0], points_transformed[:,:,0],
                                    -points_transformed[:,:,1], points_transformed[:,:,1],
                                    -points_transformed[:,:,2], points_transformed[:,:,2]], dim=-1) # BxYx6

    x = points_transformed[:, :, 0]
    y = points_transformed[:, :, 1]
    z = points_transformed[:, :, 2]
    zero = torch.FloatTensor([0]).to(x.device)
    inside_values = torch.max(torch.min(a1_ - torch.abs(x), torch.min(a2_ - torch.abs(y), a3_ - torch.abs(z))), zero) ** 2

    is_inside = torch.where(inside_values > 0,
                            torch.ones([1 for _ in range(inside_values.dim())]).to(inside_values.device),
                            torch.zeros([1 for _ in range(inside_values.dim())]).to(inside_values.device))
    is_inside_ = is_inside.unsqueeze(-1)

    outside_diffs = a_ + points_restacked

    outside_closest = torch.where(outside_diffs < 0,
                                  torch.ones([1 for _ in range(outside_diffs.dim())]).to(outside_diffs.device),
                                  torch.zeros([1 for _ in range(outside_diffs.dim())]).to(outside_diffs.device))

    inside_closest_indices = torch.min(outside_diffs, dim=-1)[1]
    inside_closest = torch.zeros_like(outside_closest).to(outside_closest.device)
    inside_closest.scatter_(-1, inside_closest_indices.unsqueeze(-1), 1)

    closest_side = torch.where(is_inside_ > 0, inside_closest, outside_closest)

    return occlusions, closest_side, distances_to_cube_sides, intersections
